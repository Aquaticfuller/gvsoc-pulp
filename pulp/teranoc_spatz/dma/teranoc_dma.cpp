/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <ips/pulp/idma_v2/be/idma_be.hpp>
#include <ips/pulp/idma_v2/be/idma_be_axi.hpp>
#include <ips/pulp/idma_v2/idma.hpp>
#include <vp/itf/io_v2.hpp>
#include <vp/vp.hpp>

#include "idma_me_dist.hpp"
#include "idma_me_split.hpp"
#include "idma_transfer_spill.hpp"
#include "teranoc_dma_frontend.hpp"

/*
 * V1-shaped distributed TeraNoC DMA with the public IOv2 single-path backend.
 *
 * The target-private stages are intentionally limited to the hardware MMIO
 * ABI and v1's split/distribute pipeline:
 *
 *   MMIO FE/spill -> RTL split -> RTL group distribution
 *                 -> RTL per-group distribution -> public BE + AXI read/write
 *
 * There is no private TCDM backend.  Both ends of each worker transfer leave
 * on its AXI pair and the owning group routes the address to L1 or externally,
 * as the RTL's post-backend AXI crossbar does.
 */
class TeranocDma : public vp::Component {
public:
  TeranocDma(vp::ComponentConf &config);

private:
  struct Worker {
    IDmaBeAxi read;
    IDmaBeAxi write;
    IDmaBe be;

    Worker(vp::Component *idma, IdmaTransferProducer *dist, int group, int dma)
        : read(idma, "axi_read_" + std::to_string(group) + "_" + std::to_string(dma), &this->be),
          write(idma, "axi_write_" + std::to_string(group) + "_" + std::to_string(dma), &this->be),
          be(idma, dist, &this->read, &this->write) {}
  };

  TeranocDmaFrontend fe;
  IDmaMeSplit me_split;
  IDmaMeDist me_cluster_dist;
  std::vector<std::unique_ptr<IDmaMeDist>> me_group_dist;
  std::vector<std::unique_ptr<IDmaTransferSpill>> wrapper_spills, group_spills;
  // deque keeps worker addresses stable after each emplacement.
  std::deque<Worker> workers;
  vp::Trace trace;
};

TeranocDma::TeranocDma(vp::ComponentConf &config)
    : vp::Component(config), fe(this, &this->me_split),
      me_split(this, &this->fe, &this->me_cluster_dist, 0),
      me_cluster_dist(this, "me_cluster_dist", &this->me_split, {}, 0) {
  this->traces.new_trace("trace", &this->trace, vp::DEBUG);
  int nb_groups = this->get_js_config()->get_int("nb_groups");
  int nb_dmas_per_group = this->get_js_config()->get_int("nb_dmas_per_group");
  int stripe_bytes = this->get_js_config()->get_int("stripe_bytes");

  if (nb_groups <= 0 || nb_dmas_per_group <= 0 || stripe_bytes <= 0 ||
      stripe_bytes % nb_dmas_per_group != 0) {
    this->trace.fatal("Invalid distributed DMA geometry " "(groups=%d, dmas/group=%d, stripe=%d)\n",
        nb_groups, nb_dmas_per_group, stripe_bytes);
  }

  int worker_region = stripe_bytes / nb_dmas_per_group;
  this->me_group_dist.reserve(nb_groups);

  std::vector<IdmaTransferConsumer *> group_itfs;
  for (int group = 0; group < nb_groups; group++) {
    IdmaTransferProducer *group_upstream = &this->me_cluster_dist;
    if (teranoc_dma_clocked()) {
      // terapool_cluster_floonoc_wrapper and mempool_group each contain a
      // request spill_register plus a registered return metadata channel.
      auto wrapper = std::make_unique<IDmaTransferSpill>(this,
          "wrapper_spill_" + std::to_string(group), &this->me_cluster_dist, nullptr);
      auto group_spill = std::make_unique<IDmaTransferSpill>(this,
          "group_spill_" + std::to_string(group), wrapper.get(), nullptr);
      wrapper->set_downstream(group_spill.get());
      group_upstream = group_spill.get();
      this->wrapper_spills.push_back(std::move(wrapper));
      this->group_spills.push_back(std::move(group_spill));
    }
    auto dist = std::make_unique<IDmaMeDist>(
        this, "me_group_dist_" + std::to_string(group), group_upstream,
        std::vector<IdmaTransferConsumer *>(), worker_region);
    IDmaMeDist *dist_ptr = dist.get();
    this->me_group_dist.push_back(std::move(dist));
    if (teranoc_dma_clocked()) {
      this->group_spills.back()->set_downstream(dist_ptr);
      group_itfs.push_back(this->wrapper_spills.back().get());
    } else {
      group_itfs.push_back(dist_ptr);
    }

    std::vector<IdmaTransferConsumer *> worker_itfs;
    for (int dma = 0; dma < nb_dmas_per_group; dma++) {
      this->workers.emplace_back(this, dist_ptr, group, dma);
      worker_itfs.push_back(&this->workers.back().be);
    }
    dist_ptr->set_be(worker_itfs);
  }

  this->me_cluster_dist.set_be(group_itfs);
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config) {
  return new TeranocDma(config);
}
