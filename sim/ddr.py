import random 
import numpy as np
import heapq
class DDRMemory:
    '''
    ---------------------------------------------------------
    DDR memory model with banks, row buffers, and latency variations
    Access latencies depend on the active bank
    Each bank has a row buffer. An access to the same row is faster. In case of
    miss, an access precharge and activate delay is added.
    
    The parameters are
    tRCD (Row to Column Delay)
    tRP (Row Precharge) : time to close one row of memory cell before opening another row
                          in the same bank.
    tCAS (Column Access Strobe latency)
    tRC (Row Cycle time)
    Behaviour:
    - The DDR requests coming from the two cores are queued and arbitrated per cycle by the DDR controler.
    - The requests are re-ordered so that the best next-command is served first:
      - prefer row hits
      - avoid accesses to busy banks
    - The bank is determined by the address' LSB: row = addr % num_banks
    - Each row contains 16 addresses: row = addr // 16
    ---------------------------------------------------------
    '''
    def __init__(self, latency=50, row_hit_latency=20, row_miss_penalty=30, precharge_time=10, num_banks=4):
        self.base_latency = latency
        self.row_hit_latency = row_hit_latency
        self.row_miss_penalty = row_miss_penalty
        self.precharge_time = precharge_time
        self.num_banks = num_banks
        self.bank_row_buffers = [None for _ in range(num_banks)]
        self.bank_precharge_end = [0 for _ in range(num_banks)]
        self.memory = {}
        self.cycle = 0

        self.pending = []  # Requests waiting to be scheduled
        self.scheduled = []  # Requests that have been scheduled for completion
        self.last_address_time = {}  # Track the latest cycle each address is scheduled to enforce order
        self.count_pps = 0

    def _get_bank(self, addr):
        return addr % self.num_banks

    def _get_row(self, addr):
        return addr // 16  # Example: each row covers 16 addresses

    # Queue a request
    def request(self, req):
        ##print(f"[Cycle {self.cycle}] ➜ New DDR request queued: Core {req.core_id}, {req.req_type.upper()} Addr {req.addr}")
        self.pending.append((self.cycle, req))

    # Process the DDR current cycle
    def tick(self):
        output = self._issue_best_request()
        #executes each instruction such that completion_time is lower than the actual time self.cycle
        while self.scheduled and self.scheduled[0][0] <= self.cycle:
            _, req = heapq.heappop(self.scheduled)
            if req.req_type == 'read':
                val = self.memory.get(req.addr, 0)
                self.count_pps+=1
                #print(self.count_pps,f"[Cycle {self.cycle}] ✔ READ COMPLETE (Core {req.core_id}, Addr {req.addr}) => {val}")
                if req.callback:
                    req.callback(val)
            elif req.req_type == 'write':
                self.count_pps+=1
                self.memory[req.addr] = req.value
                #print(self.count_pps,f"[Cycle {self.cycle}] ✔ WRITE COMPLETE (Core {req.core_id}, Addr {req.addr}) <= {req.value}")

        self.cycle += 1
        return output

    # Return the "best" request from the request queue
    # puts in self.scheduled
    def _issue_best_request(self):
        output = -1
        # Is the pending request queue empty?
        if not self.pending:
            ##print(f"[Cycle {self.cycle}] No pending DDR requests to schedule.")
            return output

        # Sort the pending requests according to their "score"
        self.pending.sort(key=lambda pair: self._score(pair[1], pair[0]))
        ##print("bank row buffer", self.bank_row_buffers)
        ##print("pending", self.pending)

        times = np.array([item[0] for item in self.pending])
        addrs = np.array([item[1].addr for item in self.pending])
        for i, (arrival_time, req) in enumerate(self.pending):
            bank = self._get_bank(req.addr)
            row = self._get_row(req.addr)
            last_time = self.last_address_time.get(req.addr, -1)

            if last_time >= self.cycle:
                continue

            if self.bank_precharge_end[bank] > self.cycle:
                continue  # Bank is busy

            if self.bank_row_buffers[bank] == row:
                delay = self.row_hit_latency
                row_status = "ROW HIT"
            else:
                delay = self.precharge_time + self.row_miss_penalty + self.row_hit_latency
                row_status = "ROW MISS"
                self.bank_row_buffers[bank] = row
                self.bank_precharge_end[bank] = self.cycle + self.precharge_time

            completion_time = self.cycle + delay
            req.completion_time = completion_time
            self.last_address_time[req.addr] = completion_time
            heapq.heappush(self.scheduled, (completion_time, req))
            self.pending.pop(i)
            min_time = min(times)
            max_time = max(times)
            max_addr = max(addrs)
            min_addr = min(addrs)
            max_radius = np.mean(np.sqrt((times - arrival_time)**2 + (addrs - req.addr)**2))
            #print(f"[Cycle {self.cycle}] ➜ Scheduling {req.req_type.upper()} for Addr {req.addr} (Core {req.core_id})")
            #print(f"                  ↳ Bank {bank}, Row {row} | {row_status} | Will complete at Cycle {completion_time}")
            return {"addr":req.addr,
                    "bank":bank,
                    "row": row,
                    "delay":delay,
                    "status":row_status, 
                    "emmission_cycle":req.num_instr, 
                    "arrival_time":arrival_time,
                    "core":req.core_id,
                    "completion_time":req.completion_time,
                    "min_time":min_time,
                    "max_time":max_time,
                    "min_addr":min_addr,
                    "max_addr":max_addr,
                    "max_radius":max_radius,
                    "pending_addr":[item[1].addr for item in self.pending],
                    "pending_core_id":[item[1].core_id for item in self.pending],
                    }
        return output

    # Compute a score for each request
    def _score(self, req, arrival_time):
        bank = self._get_bank(req.addr)
        row = self._get_row(req.addr)
        score = arrival_time

        if self.bank_row_buffers[bank] == row:
            score -= 100  # Favor row hits
        elif self.bank_precharge_end[bank] > self.cycle:
            score += 50  # Penalize busy banks

        return score
