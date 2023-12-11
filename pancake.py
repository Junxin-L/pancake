import random
import numpy as np

class pancake:
    def __init__(self, num_states, s=2, smoothing=0.0, seed=None, add_rep=True, delta=-1):
        """
        Initialize the Pancake class.

        Args:
            num_states (int): Number of states.
            smoothing (float): Smoothing parameter.
            seed (int): Random seed.
            add_rep (bool): Add replicas to keys.
            s: stage
        Attributes:
            See class attributes.

        """
        self.seed = seed
        self.smoothing = smoothing
        self.kv_num = num_states
        self.delta = delta
        self.s = s

        self.transmat = self.gen_transmat()
        self.init_probs, self.keys = self.gen_init()
        self.kv = self.cal_freq()
        self.kv_rep, self.rep_lst, self.rep_num = self.add_rep(add_rep)
        self.transmat_rep, self.transmat_rep_dict = self.re_calculate_transmat()
        self.init_probs_rep = self.re_cal_init()
        self.kv_fake = self.add_fake_access()
        self.d_lst = self.cal_d()
        self.dprime_lst = self.cal_dprime()
        self.corr_rep, self.corr = self.cal_corr()
        
        
        self.seq_lst = None
        self.access_seq_lst = None
        self.query_access_map = None

    def cal_corr(self):
        """Two stages only. Calculate correlation. Corr = dprime_1 - dprime_0"""
        corr_rep = {}
        for key in self.kv_rep.keys():
            corr_rep[key] = self.dprime_lst[1][key] - self.dprime_lst[0][key]
        corr = {}
        for key in self.kv.keys():
            corr[key] = sum([corr_rep[rep] for rep in self.rep_lst[key]])
        return corr_rep, corr
    
    def cal_d(self):
        """d_0 = init, d_t = d_{t-1} * transmat"""
        d_lst = []
        d_lst.append(self.init_probs_rep)
        for _ in range(self.s - 1):
            # Calculate frequency of each key in each stage
            # Frq(t) = Frq(t-1) * transmat
            prev_freq = d_lst[-1]
            keys = list(self.init_probs_rep.keys())

            prev_freq_values = np.array(list(prev_freq.values()))
            transmat_values = np.array(self.transmat_rep)

            # Perform matrix multiplication
            curr_freq_values = np.matmul(prev_freq_values, transmat_values)

            # Convert the result back into a dictionary
            curr_freq = dict(zip(keys, curr_freq_values))
            
            d_lst.append(curr_freq)
        self.d_lst = d_lst
        return d_lst
    
    def cal_ac_freq(self):
        """Cal appear frequency of each kv in every stage in access sequence."""
        ac_freq_lst = []
        for i in range(self.s):
            ac_freq_lst.append({})
        for seq in self.access_seq_lst:
            for i in range(self.s):
                if seq[i] not in ac_freq_lst[i].keys():
                    ac_freq_lst[i][seq[i]] = 0
                ac_freq_lst[i][seq[i]] += 1
        for i in range(self.s):
            for key in ac_freq_lst[i].keys():
                ac_freq_lst[i][key] = ac_freq_lst[i][key] / len(self.access_seq_lst)
        self.ac_freq_lst = ac_freq_lst
        return ac_freq_lst
    
    def cal_seq_freq(self):
        """Cal appear frequency of each kv in every stage in query sequence."""
        seq_freq_lst = []
        for i in range(self.s):
            seq_freq_lst.append({})
        for seq in self.seq_lst:
            for i in range(self.s):
                if seq[i] not in seq_freq_lst[i].keys():
                    seq_freq_lst[i][seq[i]] = 0
                seq_freq_lst[i][seq[i]] += 1
        for i in range(self.s):
            for key in seq_freq_lst[i].keys():
                seq_freq_lst[i][key] = seq_freq_lst[i][key] / len(self.seq_lst)
        self.seq_freq_lst = seq_freq_lst
        return seq_freq_lst
        
    
    def cal_dprime(self):
        """dp_0 = delta*d_0+(1-delta)*fake_access, dp_1=delta^2*d_1+delta*(1-delta)*d_0+(1-delta)*fake_access, only consider when s = 2"""
        dprime_lst = []
        for i in range(self.s):
            dprime_lst.append({})
        for key in self.kv_rep.keys():
            dprime_lst[0][key] = self.delta * self.init_probs_rep[key] + (1 - self.delta) * self.kv_fake[key]
        for key in self.kv_rep.keys():
            dprime_lst[1][key] = self.delta**2 * self.d_lst[1][key] + self.delta * (1 - self.delta) * self.d_lst[0][key] + (1 - self.delta) * self.kv_fake[key]
        self.dprime_lst = dprime_lst
        return dprime_lst
        
    def gen_init(self, seed=None):
        """Randomly gen init prob for all keys using seed."""
        if seed is None:    
            random.seed(seed)
        self.keys = []
        for i in range(self.kv_num):
            self.keys.append(chr(ord('a') + i))
        init_probs = {}
        for key in self.keys:
            init_probs[key] = random.random()
        # Normalize probabilities
        total = sum(init_probs.values())
        for key in init_probs.keys():
            init_probs[key] /= total
        self.init_probs = init_probs
        return init_probs, self.keys

    def gen_transmat(self):
        """Generate a random transition matrix.
    Args:
        num_states: int, number of states
        smoothing: float, smoothing parameter, lower value means more uniform distribution
        seed: int, random seed"""
        if self.seed == None:
            random.seed()
        matrix = np.random.rand(self.kv_num, self.kv_num)
        matrix /= matrix.sum(axis=1, keepdims=True)
        matrix = (1 - self.smoothing) * matrix + self.smoothing / self.kv_num
        return matrix

    def cal_freq(self):
        """Calculate frequency of each key according to init, stage and transmat."""
        stage_freq = []
        stage_freq.append(self.init_probs)
        for _ in range(self.s - 1):
            # Calculate frequency of each key in each stage
            # Frq(t) = Frq(t-1) * transmat
            prev_freq = stage_freq[-1]
            keys = list(self.init_probs.keys())

            prev_freq_values = np.array(list(prev_freq.values()))
            transmat_values = np.array(self.transmat)

            # Perform matrix multiplication
            curr_freq_values = np.matmul(prev_freq_values, transmat_values)

            # Convert the result back into a dictionary
            curr_freq = dict(zip(keys, curr_freq_values))
            
            stage_freq.append(curr_freq)
        # Calculate frequency of each key in total
        kv = {}
        for key in self.keys:
            kv[key] = sum([stage_freq[i][key] for i in range(self.s)])
        # Normalize
        total = sum(kv.values())
        for key in kv.keys():
            kv[key] /= total
        self.kv = kv
        return kv

    def key_freq(self):
        """Get frequency list using init and transmat."""
        pass

    def add_rep(self, add_rep=True):
        """Add replica to kv to make the real frequency of each key lower than alpha. Also modify rep_lst and transmat.
    """
        num = len(self.kv)
        self.kv_rep = self.kv.copy()
        self.rep_lst = {name: [] for name in self.kv_rep.keys()}
        for rep in self.rep_lst.keys():
            self.rep_lst[rep].append(rep)
        if add_rep is True:
            alpha = 1 / num
            desired_replicas = {name: int(self.kv_rep[name] / alpha) for name in self.kv_rep.keys()}
            keys_to_iterate = list(self.kv_rep.keys())
            for k in keys_to_iterate:
                i = 0
                self.kv_rep[k] = self.kv_rep[k] / (1 + desired_replicas[k])
                while i < desired_replicas[k]:
                    replica_name = f'{k}_{len(self.rep_lst[k]) + 1}'
                    self.rep_lst[k].append(replica_name)
                    self.kv_rep[replica_name] = self.kv_rep[k]
                    i += 1
        self.rep_num = sum([len(lst) for lst in self.rep_lst.values()])
        if self.delta == -1: self.delta = num / self.rep_num
        return self.kv_rep, self.rep_lst, self.rep_num

    def re_cal_init(self):
            """Re-calculate init prob after adding replicas."""
            init_probs = {}
            for key in self.kv_rep.keys():
                init_probs[key] = self.kv[key[0]] / len(self.rep_lst[key[0]])
            self.init_probs_rep = init_probs
            return init_probs

    def re_calculate_transmat(self):
        """Re-calculate the transition matrix after adding replicas. The order of the keys in values of the transition matrix is the same as the order of the keys in kv.(The order of keys are not necessarily the same as the order of the keys in rep_lst.)"""
        new_transmat_rep_dict = {}
        for key in self.kv_rep.keys():
            new_transmat_rep_dict[key] = []
            for next_key in self.kv_rep.keys():
                prob = self.transmat[ord(key[0]) - ord('a')][ord(next_key[0]) - ord('a')] / len(self.rep_lst[next_key[0]])
                new_transmat_rep_dict[key].append(prob)
        new_transmat = []
        for lst in new_transmat_rep_dict.values():
            new_transmat.append(lst)
        self.transmat_rep = new_transmat
        self.transmat_rep_dict = new_transmat_rep_dict
        return self.transmat_rep, self.transmat_rep_dict

    def add_fake_access(self):
        max_access = max(self.kv_rep.values())
        kv_fake = self.kv_rep.copy()
        for k in self.kv_rep.keys():
            kv_fake[k] = max_access - self.kv_rep[k]
        self.kv_fake = kv_fake
        # Normalize
        total = sum(kv_fake.values())
        for key in kv_fake.keys():
            kv_fake[key] /= total
        return kv_fake
    
    def add_fake_access_to_transmat(self):
        """Only used for attack. Add fake access to self.transmat."""
        fake_access_sum = {}
        for key in self.kv.keys():
            fake_access_sum[key] = 0
            for rep in self.rep_lst[key]:
                fake_access_sum[key] += self.kv_fake[rep]
        # Normalize
        for key in self.kv.keys():
            for rep in self.rep_lst[key]:
                self.kv_fake[rep] = self.kv_fake[rep] / fake_access_sum[key]
        self.transmat_with_fake_access = []
        for i, key in enumerate(self.kv.keys()):
            self.transmat_with_fake_access.append([])
            for j, next_key in enumerate(self.kv.keys()):
                if next_key == key:
                    self.transmat_with_fake_access[-1].append(self.delta*self.transmat[i][j] + (1 - self.delta) * (self.kv_fake[key] / fake_access_sum[key]))
                else:
                    self.transmat_with_fake_access[-1].append((1 - self.delta) * (self.kv_fake[key] / fake_access_sum[key]))

    def query_to_access(self):
        """Convert a query sequence to an access sequence. (The same with shuffling the query sequence))"""
        self.query_access_map = {}
        self.access_seq_lst = []
        for i, key in enumerate(self.kv_rep.keys()):
            self.query_access_map[key] = i
        for seq in self.seq_lst:
            access_seq = []
            for key in seq:
                access_seq.append(self.query_access_map[key])
            self.access_seq_lst.append(access_seq)
        return self.access_seq_lst

    def normalize_matrix(self, matrix):
        """Normalize a matrix so that each row sums to 1."""
        for i in range(len(matrix)):
            row_sum = sum(matrix[i])
            for j in range(len(matrix[i])):
                matrix[i][j] = matrix[i][j] / row_sum
        return matrix

    def normalize_dict(self, dictionary):
        total = sum(dictionary.values())
        for key in dictionary.keys():
            dictionary[key] = dictionary[key] / total
        return dictionary

    def next_query(self, d, delta):
        p = random.random()
        if p < delta:
            return random.choices(list(self.kv_rep.keys()), weights=d.values())[0], True
        else:
            return random.choices(list(self.kv_fake.keys()), weights=list(self.kv_fake.values()))[0], False

    def correlated_query_seq(self, delta=-1, count=100):
        if delta == -1:
            delta = self.delta
        seq_lst = []
        for _ in range(count):
            d_lst = enumerate(self.d_lst)
            seq = []
            if self.seed is None:
                random.seed()
            d = next(d_lst)[1]
            query, is_real = self.next_query(d, delta)
            seq.append(query)
            for _ in range(self.s - 1):
                if is_real:
                    d = next(d_lst)[1]
                query, is_real = self.next_query(d, delta)
                seq.append(query)
            seq_lst.append(seq)
        self.seq_lst = seq_lst
        self.query_to_access()
        return seq_lst
    
    def rank_seq_freq(self):
        """Observe the substraction of frequency of each key in two stages and sort by the result."""
        seq_freq_lst = self.cal_seq_freq()
        seq_freq_diff = {}
        for key in self.kv_rep.keys():
            seq_freq_diff[key] = seq_freq_lst[1][key] - seq_freq_lst[0][key]
        seq_freq_diff = sorted(seq_freq_diff.items(), key=lambda x: x[1], reverse=True)
        return seq_freq_diff
    
    def rank_ac_freq(self):
        """Observe the substraction of frequency of each key in two stages and sort by the result."""
        ac_freq_lst = self.cal_ac_freq()
        ac_freq_diff = {}
        for key in self.query_access_map.values():
            ac_freq_diff[key] = ac_freq_lst[1][key] - ac_freq_lst[0][key]
        ac_freq_diff = sorted(ac_freq_diff.items(), key=lambda x: x[1], reverse=True)
        return ac_freq_diff

    def rank_corr(corr):
        """Sort correlation in descending order."""
        corr = sorted(corr.items(), key=lambda x: x[1], reverse=True)
        return corr
    
if __name__ == '__main__':
    pass