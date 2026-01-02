TEAM_100_IDS = {1, 2, 3, 4, 5}
TEAM_200_IDS = {6, 7, 8, 9, 10}


def extract_team_features(timeline, team_id: int, minute: int = 15):
    """Zwraca uśrednione statystyki dla konkretnej drużyny w zadanej minucie."""
    frames = timeline["info"]["frames"]
    frame = frames[-1] if minute >= len(frames) else frames[minute]

    team_ids = TEAM_100_IDS if team_id == 100 else TEAM_200_IDS
    enemy_ids = TEAM_200_IDS if team_id == 100 else TEAM_100_IDS

    stats = {
        "gold": 0,
        "gold_per_sec": 0,
        "cs": 0,
        "jungle_cs": 0,
        "level": 0,
        "xp": 0,
        "total_damage_done": 0,
        "total_damage_taken": 0,
        "damage_to_champions": 0,
        "wards_placed": 0,
        "wards_killed": 0,
        "vision_score": 0,
    }

    events = {
        "kills": 0,
        "deaths": 0,
        "assists": 0,
        "towers": 0,
        "dragons": 0,
        "herald": 0,
        "first_blood": 0,
        "first_tower": 0,
        "first_dragon": 0,
        "first_herald": 0,
    }

    for participant in frame["participantFrames"].values():
        pid = participant["participantId"]
        if pid not in team_ids:
            continue
        stats["gold"] += participant["totalGold"]
        stats["gold_per_sec"] += participant.get("goldPerSecond", 0)
        stats["cs"] += participant["minionsKilled"]
        stats["jungle_cs"] += participant["jungleMinionsKilled"]
        stats["level"] += participant["level"]
        stats["xp"] += participant["xp"]
        stats["total_damage_done"] += participant["damageStats"]["totalDamageDone"]
        stats["total_damage_taken"] += participant["damageStats"]["totalDamageTaken"]
        stats["damage_to_champions"] += participant["damageStats"]["totalDamageDoneToChampions"]
        stats["wards_placed"] += participant.get("wardsPlaced", 0)
        stats["wards_killed"] += participant.get("wardsKilled", 0)
        stats["vision_score"] += participant.get("visionScore", 0)

    first_blood_done = False
    first_tower_done = False
    first_dragon_done = False
    first_herald_done = False

    for frame_entry in frames[: minute + 1]:
        for event in frame_entry["events"]:
            etype = event.get("type")
            if etype == "CHAMPION_KILL":
                killer = event.get("killerId")
                victim = event.get("victimId")
                assists = event.get("assistingParticipantIds", [])

                if killer in team_ids:
                    events["kills"] += 1
                    if not first_blood_done:
                        events["first_blood"] = 1
                        first_blood_done = True
                if victim in team_ids:
                    events["deaths"] += 1
                events["assists"] += sum(1 for aid in assists if aid in team_ids)

            elif etype == "BUILDING_KILL":
                if event.get("teamId") == team_id:
                    events["towers"] += 1
                    if not first_tower_done:
                        events["first_tower"] = 1
                        first_tower_done = True

            elif etype == "ELITE_MONSTER_KILL":
                if event.get("killerId") in team_ids:
                    if event.get("monsterType") == "DRAGON":
                        events["dragons"] += 1
                        if not first_dragon_done:
                            events["first_dragon"] = 1
                            first_dragon_done = True
                    if event.get("monsterType") == "RIFTHERALD":
                        events["herald"] += 1
                        if not first_herald_done:
                            events["first_herald"] = 1
                            first_herald_done = True

            # Jeśli pierwszy krew padła na wrogiej drużynie, ustaw flagę by unikać podwójnego oznaczenia
            if etype == "CHAMPION_KILL" and not first_blood_done and killer in enemy_ids:
                first_blood_done = True

    players_count = len(team_ids)
    features = {f"{k}_avg": v / players_count for k, v in stats.items()}
    features.update({
        "kills_avg": events["kills"] / players_count,
        "deaths_avg": events["deaths"] / players_count,
        "assists_avg": events["assists"] / players_count,
        "towers": events["towers"],
        "dragons": events["dragons"],
        "herald": events["herald"],
        "first_blood": events["first_blood"],
        "first_tower": events["first_tower"],
        "first_dragon": events["first_dragon"],
        "first_herald": events["first_herald"],
    })

    return features


def extract_features(timeline, minute: int = 15):
    """Kompatybilność wsteczna: różnice statystyk między drużynami."""
    t100 = extract_team_features(timeline, 100, minute)
    t200 = extract_team_features(timeline, 200, minute)

    features = {}
    for key in t100:
        features[f"{key}_diff"] = t100[key] - t200[key]
    return features

