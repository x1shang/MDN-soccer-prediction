"""Screen DFL ball tracks; outputs candidates, not proven contact-free flights."""
import csv, hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'B_standard_ball_trajectories' / 'processed'
MATCHES = ['J03WMX', 'J03WOH', 'J03WR9']
MIN_Z, MIN_FRAMES, MAX_FRAMES, SEPARATION = 0.5, 16, 25, 125
MAX_SPEED, MAX_DV = 40.0, 1.6  # m/s; per 0.04 s change, for gross anomaly screening only

def dist(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def main():
    all_rows, manifest, sources = [], [], []
    for match in MATCHES:
        file = SRC / f'{match}_secondHalf_ball_native.csv'
        sources.append({'match':match,'local_source':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        with file.open(encoding='utf-8-sig', newline='') as f:
            records = list(csv.DictReader(f))
        run, candidates = [], []
        def flush():
            nonlocal run
            if len(run) >= MIN_FRAMES:
                candidates.append(run)
            run = []
        for rec in records:
            try:
                frame=int(rec['frame_id']); xyz=tuple(float(rec[k]) for k in ('x_m','y_m','z_m'))
                ok=(rec['ball_status']=='1' and all(math.isfinite(v) for v in xyz) and xyz[2]>=MIN_Z)
                if not ok:
                    flush(); continue
                point=(frame,xyz,rec['timestamp_utc'])
                if run:
                    v=tuple((xyz[i]-run[-1][1][i])/0.04 for i in range(3))
                    if frame != run[-1][0]+1 or math.sqrt(sum(t*t for t in v))>MAX_SPEED:
                        flush()
                    elif len(run)>1:
                        prev=tuple((run[-1][1][i]-run[-2][1][i])/0.04 for i in range(3))
                        if dist(v,prev)>MAX_DV:
                            flush()
                run.append(point)
            except (ValueError, KeyError):
                flush()
        flush()
        last_end=-100000
        for segment in candidates:
            if segment[0][0]-last_end < SEPARATION: continue
            # Require net movement, and take one bounded window per candidate run.
            window=segment[:MAX_FRAMES]
            if dist(window[0][1],window[-1][1]) < 4: continue
            cid=f'{match}_{window[0][0]}_{window[-1][0]}'
            if (match=='J03WOH' and window[0][0]<=128965 and window[-1][0]>=128955) or (match=='J03WR9' and window[0][0]<=176254 and window[-1][0]>=176244): continue
            last_end=window[-1][0]
            for n,(frame,xyz,ts) in enumerate(window):
                all_rows.append({'case_id':cid,'match_id':match,'frame_id':frame,'timestamp_utc':ts,'t_s':round(n*.04,2),'x_m':xyz[0],'y_m':xyz[1],'z_m':xyz[2],'role':'input' if n<3 else 'held_out'})
            manifest.append({'case_id':cid,'frame_start':window[0][0],'frame_end':window[-1][0],'n_frames':len(window),'duration_s':round((len(window)-1)*.04,2),'min_z_m':min(p[1][2] for p in window),'max_z_m':max(p[1][2] for p in window),'displacement_m':round(dist(window[0][1],window[-1][1]),3),'contact_status':'unknown','event_type':'unknown'})
            if sum(1 for m in manifest if m['case_id'].startswith(match+'_'))>=20: break
    for name,data in [('positions_candidates.csv',all_rows),('segments_candidates.csv',manifest)]:
        with (OUT/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (OUT/'source_hashes.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('segments',len(manifest),'points',len(all_rows),'by match', {m:sum(x['case_id'].startswith(m+'_') for x in manifest) for m in MATCHES})

if __name__=='__main__': main()

