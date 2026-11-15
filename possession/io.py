"""Read the provided match assets into numpy arrays.

Everything is indexed by tactical *video frame* (0-based), which is also the submission row.
The ATD counts frames from 1, so ``video_frame = frame_id - 1``.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

STEM = "BAR-ATH_tactical"
N_FRAMES = 199319
FPS = 25
# Pitch size in metres (challenge repository, games.py).
PITCH = (105.3, 68.0)
# Graded blocks, video frames, inclusive (challenge repository, DATA.md / games.py).
HALVES = ((11185, 90651), (110808, 190197))

HOME, AWAY = 0, 1


def paths(data_dir):
    folder = os.path.join(data_dir, "tactical")
    return {
        "atd_meta": os.path.join(folder, STEM + "_atd.xml"),
        "atd_raw": os.path.join(folder, STEM + "_atd.txt"),
        "smart_tagging": os.path.join(folder, STEM + "_smart_tagging.xml"),
        "homography": os.path.join(folder, STEM + "_homography.csv"),
    }


def slot_teams(meta_path):
    """Team of every track slot (0 home, 1 away) and the two club names, home first.

    The first team in the EPTS metadata is the home side.
    """
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]
    slots = np.array([HOME if player.get("teamId") == home_id else AWAY
                      for player in root.iter("Player")], dtype=np.int8)
    return slots, (teams[0][1], teams[1][1])


def home_attacks_right_first_half(meta_path):
    """True if the home side attacks towards x = 1 in the first half (EPTS metadata)."""
    root = ET.parse(meta_path).getroot()
    for parameter in root.iter("ProviderParameter"):
        if parameter.findtext("Name") == "attack_direction_first_half":
            return (parameter.findtext("Value") or "").strip().lower() == "left_to_right"
    return True


def load_atd(data_dir, cache_dir=None):
    """Tracking as arrays in metres.

    Returns a dict:
      px, py   (N, S) float32  player positions in metres, NaN where the slot is empty
      ps       (N, S) float32  player speed, m/s, as provided
      bx, by   (N,)   float32  ball position in metres, NaN where it is not detected
      team     (S,)   int8     0 home, 1 away
      names    (home, away)
    Parsing the 190 MB text file takes about a minute, so the result is cached as .npz.
    """
    p = paths(data_dir)
    cache = os.path.join(cache_dir, "atd.npz") if cache_dir else None
    team, names = slot_teams(p["atd_meta"])
    if cache and os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(p["atd_raw"]):
        z = np.load(cache)
        return {"px": z["px"], "py": z["py"], "ps": z["ps"], "bx": z["bx"], "by": z["by"],
                "team": team, "names": names}

    n_slots = len(team)
    width = 1 + 3 * n_slots + 3
    table = np.full((N_FRAMES, width), np.nan, dtype=np.float32)
    with open(p["atd_raw"]) as handle:
        for line in handle:
            values = line.strip().replace(":", ",").replace(";", ",").split(",")
            if len(values) != width:
                raise ValueError("unexpected ATD line with {} fields".format(len(values)))
            row = np.array(values, dtype=np.float32)
            table[int(row[0]) - 1] = row

    players = table[:, 1:1 + 3 * n_slots].reshape(N_FRAMES, n_slots, 3)
    out = {
        "px": np.ascontiguousarray(players[:, :, 0]) * PITCH[0],
        "py": np.ascontiguousarray(players[:, :, 1]) * PITCH[1],
        "ps": np.ascontiguousarray(players[:, :, 2]),
        "bx": table[:, -3] * PITCH[0],
        "by": table[:, -2] * PITCH[1],
    }
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, **out)
    out["team"] = team
    out["names"] = names
    return out


def load_smart_tagging(data_dir, names):
    """Smart Tagging instances as dicts with video-frame intervals.

    ``team`` is 0 (home), 1 (away) or -1 where the file says N/A or names neither club.
    """
    out = []
    home = names[0].strip().lower()
    away = names[1].strip().lower()
    for instance in ET.parse(paths(data_dir)["smart_tagging"]).iter("instance"):
        tags = {label.findtext("group"): (label.findtext("text") or "")
                for label in instance.findall("label")}
        club = (tags.get("Team") or "").strip().lower()
        out.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start": int(round(float(instance.findtext("start")) * FPS)),
            "end": int(round(float(instance.findtext("end")) * FPS)),
            "team": HOME if club == home else AWAY if club == away else -1,
            "type": (tags.get("Type") or "").strip().lower(),
            "tags": tags,
        })
    return out


def load_target(data_dir, cache_dir=None):
    """The challenge match as a match dict (see model.py): tracking, layout and Smart Tagging."""
    data = load_atd(data_dir, cache_dir)
    return {
        "name": "BAR-ATH", "px": data["px"], "py": data["py"], "bx": data["bx"], "by": data["by"],
        "team": data["team"], "names": data["names"], "n_frames": N_FRAMES,
        "halves": list(HALVES), "tags": load_smart_tagging(data_dir, data["names"]),
    }


def write_submission(path, labels):
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path
