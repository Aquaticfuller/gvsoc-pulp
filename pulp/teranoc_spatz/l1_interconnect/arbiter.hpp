/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

/*
 * The arbiter behind every contended port of the TeraNoC L1 fabric.
 *
 * RTL uses one cell everywhere: `rr_arb_tree` (common_cells). The NoC router
 * outputs reach it through `floo_wormhole_arbiter`, the crossbars and the TCDM
 * bank ports through `stream_xbar`, and both instantiate it the same way --
 * `ExtPrio=0, AxiVldRdy=1, LockIn=1` (stream_xbar.sv:28-37,
 * floo_wormhole_arbiter.sv:35-41) and `FairArb=1` (rr_arb_tree.sv:80). So
 * ArbPolicy::RrArbTree is the hardware; the other policies are model experiments.
 *
 * Index space: the pick depends on the NUMERIC index, so a call site MUST number
 * its ports the way RTL does. There is no permutation argument on purpose.
 */

#pragma once

#include <cstdint>

enum class ArbPolicy {
    // The RTL cell: smallest (input ^ rr_q), FairArb pointer update.
    RrArbTree,
    // The simplified scan the model used before: first contender at or after
    // rr_q in cyclic order, pointer to winner + 1.
    CyclicRr,
    // Lowest index always wins. No state.
    FixedPriority,
};

class Arbiter {
public:
    // nb_inputs <= 64. Indices are the caller's port indices.
    void init(int nb_inputs, ArbPolicy policy = ArbPolicy::RrArbTree, bool lock_in = true) {
        this->nb_inputs = nb_inputs;
        this->policy = policy;
        this->lock_in = lock_in;
        this->reset();
    }

    // rr_arb_tree resets rr_q to RTL input 0.
    void reset() {
        this->rr = 0;
        this->locked_requests = 0;
        this->locked = false;
    }

    // Contenders, as a mask of port indices. Returns the winning port index, or
    // -1 when nothing contends. A locked arbiter ignores the mask it is handed.
    int select(uint64_t requests) {
        uint64_t req = requests;

        if (this->lock_in) {
            if (this->locked) {
                req = this->locked_requests;
            } else if (req != 0) {
                this->locked_requests = req;
                this->locked = true;
            }
        } else if (req != 0) {
            // Without LockIn the mask is rebuilt every cycle, but it must still
            // survive from the select that elected a transfer to the grant that
            // retires it, or FairArb advances over an empty mask.
            this->locked_requests = req;
        }

        if (req == 0) {
            return -1;
        }

        this->last_winner = this->pick(req);
        return this->last_winner;
    }

    // The frozen selection while a grant is outstanding, else -1.
    int locked_winner() const { return this->locked ? this->last_winner : -1; }

    // The grant was TAKEN -- gnt_i pulsed. Advances the pointer and releases the
    // lock. Call it once per grant, not once per flit of a wormhole packet.
    void grant(int winner) {
        switch (this->policy) {
        case ArbPolicy::RrArbTree:
            this->rr = this->next_fair(this->locked_requests);
            break;
        case ArbPolicy::CyclicRr:
            this->rr = (winner + 1) % this->nb_inputs;
            break;
        case ArbPolicy::FixedPriority:
            break;
        }

        this->locked_requests = 0;
        this->locked = false;
    }

    // For probes and traces.
    int width() const { return this->nb_inputs; }
    int pointer() const { return this->rr; }
    bool is_locked() const { return this->locked; }

private:
    // rr_arb_tree takes the side rr_q points at whenever both sides of a node
    // request and the other side otherwise, which lexicographically minimises
    // the bits of (input ^ rr_q).
    int pick(uint64_t requests) const {
        int winner = -1;
        int winner_rank = 0;
        for (int input = 0; input < this->nb_inputs; input++) {
            if (!(requests & (1ULL << input))) {
                continue;
            }
            if (this->policy == ArbPolicy::FixedPriority) {
                return input;
            }
            int rank = this->policy == ArbPolicy::RrArbTree
                ? (input ^ this->rr)
                : ((input - this->rr + this->nb_inputs) % this->nb_inputs);
            if (winner == -1 || rank < winner_rank) {
                winner = input;
                winner_rank = rank;
            }
        }
        return winner;
    }

    // FairArb: lowest set index above rr_q, wrapping round to the lowest at or
    // below it (rr_arb_tree.sv:198-199 upper_mask / lower_mask).
    int next_fair(uint64_t requests) const {
        for (int step = 1; step <= this->nb_inputs; step++) {
            int input = (this->rr + step) % this->nb_inputs;
            if (requests & (1ULL << input)) {
                return input;
            }
        }
        return this->rr;
    }

    int nb_inputs = 0;
    ArbPolicy policy = ArbPolicy::RrArbTree;
    bool lock_in = true;

    // rr_q, and the frozen req_d plus its lock_q.
    int rr = 0;
    uint64_t locked_requests = 0;
    bool locked = false;
    // Last pick. Only meaningful while locked.
    int last_winner = -1;
};
