# schedulers/fifo_exclusive.py
'''
class FIFOExclusive:
    def __init__(self): pass

    def pick_next(self, queue, usage_dict):
        if not queue: return None
        queue.sort(key=lambda x: x["submit_ts"])
        return queue[0]
'''

class FIFOExclusive:
    def __init__(self): pass

    def pick_next(self, queue, last_submit):
        if not queue: return None
        # 忽略任何 state 限制，照提交順序選
        queue.sort(key=lambda x: last_submit.get(x["jobid"], x["submit_ts"]))
        return queue[0]

