# schedulers/timeslice_maxmin.py
class TimesliceMaxMin:
    def __init__(self, slice_sec=900):
        self.slice_sec = slice_sec

    def pick_next(self, queue, usage_dict):
        if not queue: return None
        users = {}
        for j in queue:
            users.setdefault(j["user_id"], 0)
        # 把未出現的 user 也設 0
        for u in list(users.keys()):
            users[u] = usage_dict.get(u, 0)
        # 找目前用最少的 user
        target_user = min(users.keys(), key=lambda u: users[u])
        # 該使用者取最老的 job
        cand = [j for j in queue if j["user_id"] == target_user]
        cand.sort(key=lambda x: x["submit_ts"])
        return cand[0] if cand else None

