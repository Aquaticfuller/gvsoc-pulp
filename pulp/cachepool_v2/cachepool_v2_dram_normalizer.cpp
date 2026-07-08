/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Pass-through shim for Spatz VLSU ports.
 *
 * Previously subtracted 0x20000000 from 0xa0000000+ addresses to "normalize"
 * the uncached DRAM region. That was wrong: the ELF loader places .pdcp_src
 * data at VMA 0xa0000000, so the cache refill must also go to 0xa0000000 (which
 * the SoC AXI router forwards to pdcp_mem). Normalization caused refills to go
 * to 0x80000000 (l2_mem / code), returning garbage instead of data.
 *
 * The shim is kept because it uses req_forward (zero IoReq arg slots) instead
 * of a Router (arg_alloc(4) per traversal), avoiding arg-stack overflow at the
 * FlooNoc NI.
 */

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>

class DramNormalizer : public vp::Component
{
public:
    DramNormalizer(vp::ComponentConf &config);

private:
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static void grant(vp::Block *__this, vp::IoReq *req);
    static void response(vp::Block *__this, vp::IoReq *req);

    vp::Trace    trace;
    vp::IoSlave  input_itf;
    vp::IoMaster output_itf;
};

DramNormalizer::DramNormalizer(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->input_itf.set_req_meth(&DramNormalizer::req);
    this->output_itf.set_resp_meth(&DramNormalizer::response);
    this->output_itf.set_grant_meth(&DramNormalizer::grant);
    this->new_slave_port("input",  &this->input_itf);
    this->new_master_port("output", &this->output_itf);
}

vp::IoReqStatus DramNormalizer::req(vp::Block *__this, vp::IoReq *req)
{
    DramNormalizer *_this = (DramNormalizer *)__this;
    return _this->output_itf.req_forward(req);
}

void DramNormalizer::grant(vp::Block *__this, vp::IoReq *req)
{
    DramNormalizer *_this = (DramNormalizer *)__this;
    _this->input_itf.grant(req);
}

void DramNormalizer::response(vp::Block *__this, vp::IoReq *req)
{
    DramNormalizer *_this = (DramNormalizer *)__this;
    _this->input_itf.resp(req);
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new DramNormalizer(config);
}
