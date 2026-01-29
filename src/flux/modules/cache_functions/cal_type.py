from .force_scheduler import force_scheduler


def cal_type(cache_dic, current):
    """
    Determine calculation type for this step
    """
    # 🔥 Add debug info: print config info at step 0-2
    # if current['step'] <= 2:
    #     debug_info = (
    #         f"[CACHE DEBUG] Step {current['step']}: "
    #         f"fresh_ratio={cache_dic['fresh_ratio']}, "
    #         f"taylor_cache={cache_dic['taylor_cache']}, "
    #         f"fresh_threshold={cache_dic['fresh_threshold']}, "
    #         f"first_enhance={cache_dic.get('first_enhance', 'N/A')}"
    #     )
    #     print(debug_info)

    # 🔥 New: collect mode - specifically for feature collection
    if cache_dic.get("collect_mode", False):
        current["type"] = "full"
        if current["step"] == 0:
            current["activated_steps"].append(current["step"])
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: Select COLLECT mode -> type='full' (feature collection)")
        return

    # Hard guard: original mode should be full at every step regardless of interval/first_enhance settings
    if cache_dic.get("mode") == "original":
        current["type"] = "full"
        if current["step"] == 0:
            current["activated_steps"].append(current["step"])
        return

    # 🔥 Fix: In original mode, all steps should be 'full' type
    if (
        (cache_dic["fresh_ratio"] == 0.0)
        and (not cache_dic["taylor_cache"])
        and (cache_dic["fresh_threshold"] == 1)
    ):
        # Original mode: Every step performs full computation, no caching
        current["type"] = "full"
        if current["step"] == 0:
            current["activated_steps"].append(current["step"])
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: Select ORIGINAL mode -> type='full'")
        return

    # 🔥 Fix: Correctly implement first_enhance logic
    # All steps during first_enhance period should be full mode
    in_first_enhance_period = current["step"] < cache_dic["first_enhance"]

    if (cache_dic["fresh_ratio"] == 0.0) and (not cache_dic["taylor_cache"]):
        # FORA/Delta: default behavior is only step-0 full, but for Delta we want to respect
        # `first_enhance` as a warm-up period (so users can trade speed for quality).
        if cache_dic.get("Delta-DiT", False):
            first_step = in_first_enhance_period
        else:
            first_step = current["step"] == 0
    else:
        # ToCa/Taylor/HiCache: first `first_enhance` steps are full.
        first_step = in_first_enhance_period

    force_fresh = cache_dic["force_fresh"]
    if not first_step:
        fresh_interval = cache_dic["cal_threshold"]
    else:
        fresh_interval = cache_dic["fresh_threshold"]

    if (first_step) or (cache_dic["cache_counter"] == fresh_interval - 1):
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        # current['activated_times'].append(current['t'])
        force_scheduler(cache_dic, current)
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: first_step={first_step}, 选择 -> type='full'")

    elif cache_dic["taylor_cache"]:
        cache_dic["cache_counter"] += 1
        if cache_dic.get("cluster_num", 0) > 0:
            current["type"] = "ClusCa"
        else:
            current["type"] = "taylor_cache"
        # if current['step'] <= 2:
        #     print(f"[CACHE DEBUG] Step {current['step']}: 选择 TAYLOR_CACHE 模式 -> type='taylor_cache'")

    # Delta-DiT: only alternate between full steps and delta-reuse steps.
    # IMPORTANT: do not fall back to ToCa's parity-based schedule (it requires token caches).
    elif cache_dic["Delta-DiT"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "Delta-Cache"

    elif cache_dic["cache_counter"] % 2 == 1:  # 0: ToCa-Aggresive-ToCa, 1: Aggresive-ToCa-Aggresive
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
    # 'cache_noise' 'ToCa' 'FORA'
    else:
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
        # if current['step'] < 25:
        #    current['type'] = 'FORA'
        # else:
        #    current['type'] = 'aggressive'

    accelerator = cache_dic.get("eigencache_accel")
    if accelerator:
        if accelerator.should_force_full(current):
            if current["type"] != "full":
                current["type"] = "full"
                if not current["activated_steps"] or current["activated_steps"][-1] != current["step"]:
                    current["activated_steps"].append(current["step"])
                cache_dic["cache_counter"] = 0
                force_scheduler(cache_dic, current)


######################################################################
# if (current['step'] in [3,2,1,0]):
#    current['type'] = 'full'
