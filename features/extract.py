def extract_features(timeline, minute=15):
    frames = timeline["info"]["frames"]
    # jeśli mecz krótszy niż 15 minut
    if minute >= len(frames):
        frame = frames[-1]
    else:
        frame = frames[minute]

    team100_ids = [1, 2, 3, 4, 5]
    team200_ids = [6, 7, 8, 9, 10]

    # ====== Team stats ======
    t100_stats = {"gold":0,"gold_per_sec":0,"cs":0,"jungle_cs":0,"level":0,"xp":0,
                  "total_damage_done":0,"total_damage_taken":0,"damage_to_champions":0,
                  "wards_placed":0,"wards_killed":0,"vision_score":0}
    t200_stats = t100_stats.copy()

    for p in frame["participantFrames"].values():
        pid = p["participantId"]
        team = t100_stats if pid in team100_ids else t200_stats
        team["gold"] += p["totalGold"]
        team["gold_per_sec"] += p.get("goldPerSecond",0)
        team["cs"] += p["minionsKilled"]
        team["jungle_cs"] += p["jungleMinionsKilled"]
        team["level"] += p["level"]
        team["xp"] += p["xp"]
        # obrażenia
        team["total_damage_done"] += p["damageStats"]["totalDamageDone"]
        team["total_damage_taken"] += p["damageStats"]["totalDamageTaken"]
        team["damage_to_champions"] += p["damageStats"]["totalDamageDoneToChampions"]
        # wizja
        team["wards_placed"] += p.get("wardsPlaced",0)
        team["wards_killed"] += p.get("wardsKilled",0)
        team["vision_score"] += p.get("visionScore",0)

    # ====== Events ======
    t100_events = {"kills":0,"deaths":0,"assists":0,"towers":0,"dragons":0,"herald":0,
                   "first_blood":0,"first_tower":0,"first_dragon":0,"first_herald":0}
    t200_events = t100_events.copy()

    first_blood_done = False
    first_tower_done = False
    first_dragon_done = False
    first_herald_done = False

    for f in frames[:minute+1]:
        for e in f["events"]:
            pid = e.get("killerId")
            # ===== CHAMPION KILL =====
            if e["type"] == "CHAMPION_KILL":
                if pid in team100_ids:
                    t100_events["kills"] += 1
                    t200_events["deaths"] += 1
                    if not first_blood_done:
                        t100_events["first_blood"] = 1
                        first_blood_done = True
                elif pid in team200_ids:
                    t200_events["kills"] += 1
                    t100_events["deaths"] += 1
                    if not first_blood_done:
                        t200_events["first_blood"] = 1
                        first_blood_done = True
            # ===== BUILDING KILL =====
            if e["type"] == "BUILDING_KILL":
                team_id = 100 if pid in team100_ids else 200
                if e["teamId"] == 100:
                    t100_events["towers"] += 1
                    if not first_tower_done:
                        t100_events["first_tower"] = 1
                        first_tower_done = True
                else:
                    t200_events["towers"] += 1
                    if not first_tower_done:
                        t200_events["first_tower"] = 1
                        first_tower_done = True
            # ===== ELITE MONSTER =====
            if e["type"] == "ELITE_MONSTER_KILL":
                if pid in team100_ids:
                    if e["monsterType"] == "DRAGON":
                        t100_events["dragons"] += 1
                        if not first_dragon_done:
                            t100_events["first_dragon"] = 1
                            first_dragon_done = True
                    if e["monsterType"] == "RIFTHERALD":
                        t100_events["herald"] += 1
                        if not first_herald_done:
                            t100_events["first_herald"] = 1
                            first_herald_done = True
                elif pid in team200_ids:
                    if e["monsterType"] == "DRAGON":
                        t200_events["dragons"] += 1
                        if not first_dragon_done:
                            t200_events["first_dragon"] = 1
                            first_dragon_done = True
                    if e["monsterType"] == "RIFTHERALD":
                        t200_events["herald"] += 1
                        if not first_herald_done:
                            t200_events["first_herald"] = 1
                            first_herald_done = True

    # ====== Łączenie w features ======
    features = {}
    for k in t100_stats:
        features[f"{k}_diff"] = t100_stats[k] - t200_stats[k]
    for k in t100_events:
        features[f"{k}_diff"] = t100_events[k] - t200_events[k]

    return features

