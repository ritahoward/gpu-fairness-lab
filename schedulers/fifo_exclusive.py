# schedulers/fifo_exclusive.py
class FIFOExclusive:
    def __init__(self): pass

    def pick_next(self, queue, usage_dict):
        if not queue: return None
        queue.sort(key=lambda x: x["submit_ts"])
        return queue[0]

