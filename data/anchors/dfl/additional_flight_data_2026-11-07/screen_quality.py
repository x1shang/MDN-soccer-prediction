"""Screen full tracks for gross anomalies; survivors are NOT contact-verified."""
import csv, math, xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

P = Path(__file__).resolve().parent
EVENT_DIR = P.parent / 'B_standard_ball_trajectories' / 'raw' / 'dfl'
CONTACT_TAGS = {'Play','OtherBallAction','TacklingGame','ThrowIn','BallClaiming','FreeKick','ShotAtGoal','GoalKick','CornerKick','KickOff','BallDeflection','Penalty','RefereeBall'}

def read(name):
    with (P/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows, fields):
    with (P/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def norm(v): return math.sqrt(sum(x*x for x in v))

def solve(m, b):
    a=[list(row)+[value] for row,value in zip(m,b)]
    for k in range(3):
        pivot=max(range(k,3),key=lambda i:abs(a[i][k])); a[k],a[pivot]=a[pivot],a[k]
        div=a[k][k]
        for j in range(k,4): a[k][j]/=div
        for i in range(3):
            if i==k: continue
            factor=a[i][k]
            for j in range(k,4): a[i][j]-=factor*a[k][j]
    return [a[i][3] for i in range(3)]

def main():
    positions, segments=read('positions_candidates.csv'),read('segments_candidates.csv')
    groups=defaultdict(list)
    for row in positions: groups[row['case_id']].append(row)
    events={}
    for match in sorted({r['match_id'] for r in positions}):
        paths=list(EVENT_DIR.glob(f'*events*{match}.xml'))
        if len(paths)!=1: raise RuntimeError(f'Expected exactly one event XML for {match}, got {paths}')
        events[match]=[]
        for event in ET.parse(paths[0]).getroot().iter('Event'):
            kinds=sorted(CONTACT_TAGS.intersection(child.tag for child in event))
            if kinds and event.get('EventTime'):
                events[match].append((datetime.fromisoformat(event.get('EventTime')),event.get('EventId',''),'+'.join(kinds)))
    audit=[]
    for seg in segments:
        rows=groups[seg['case_id']]
        t=[float(r['t_s']) for r in rows]; p=[[float(r[k]) for k in ('x_m','y_m','z_m')] for r in rows]
        frames=[int(r['frame_id']) for r in rows]
        times=[datetime.fromisoformat(r['timestamp_utc']) for r in rows]
        flags=[]
        if len(rows)<16 or len(rows)!=int(seg['n_frames']) or any(b-a!=1 for a,b in zip(frames,frames[1:])) or any(abs((b-a).total_seconds()-.04)>1e-6 for a,b in zip(times,times[1:])) or any(abs(x-i*.04)>1e-8 for i,x in enumerate(t)): flags.append('time_or_count')
        if not all(math.isfinite(x) for row in p for x in row) or min(row[2] for row in p)<.5: flags.append('invalid_or_low')
        speed=[norm([(b[j]-a[j])/.04 for j in range(3)]) for a,b in zip(p,p[1:])]
        if min(speed)<3 or max(speed)>40: flags.append('speed_outside_3_40')
        # Whole-track quadratic is a geometry quality diagnostic, not a prediction.
        A=[[1,x,x*x/2] for x in t]
        ata=[[sum(row[i]*row[j] for row in A) for j in range(3)] for i in range(3)]
        c=[solve(ata,[sum(A[k][j]*p[k][axis] for k in range(len(t))) for j in range(3)]) for axis in range(3)]
        residual=[norm([p[k][i]-sum(A[k][j]*c[i][j] for j in range(3)) for i in range(3)]) for k in range(len(t))]
        a=[c[i][2] for i in range(3)]; rmse=math.sqrt(sum(x*x for x in residual)/len(residual)); peak=max(residual)
        if rmse>.08 or peak>.18: flags.append('irregular_track')
        if not (-20<=a[2]<=-1): flags.append('vertical_accel_outside_screen')
        if norm(a[:2])>15: flags.append('horizontal_accel_outside_screen')
        nearby=[(dt,eid,kind) for dt,eid,kind in events[rows[0]['match_id']] if -1 <= (dt-times[0]).total_seconds() <= (times[-1]-times[0]).total_seconds()+1]
        if nearby: flags.append('ball_event_within_1s_buffer')
        audit.append({'case_id':seg['case_id'],'decision':'exclude' if flags else 'provisional','reasons':';'.join(flags),'nearby_events':'|'.join(f'{dt.isoformat()}:{eid}:{kind}' for dt,eid,kind in nearby),'n_frames':len(rows),'rmse_3d_m':round(rmse,5),'max_residual_m':round(peak,5),'accel_x_m_s2':round(a[0],3),'accel_y_m_s2':round(a[1],3),'accel_z_m_s2':round(a[2],3),'min_step_speed_m_s':round(min(speed),3),'max_step_speed_m_s':round(max(speed),3)})
    keep={r['case_id'] for r in audit if r['decision']=='provisional'}
    write('screening_audit.csv',audit,list(audit[0]))
    write('positions.csv',[r for r in positions if r['case_id'] in keep],list(positions[0]))
    write('segments.csv',[r for r in segments if r['case_id'] in keep],list(segments[0]))
    print('counts:',Counter(r['decision'] for r in audit),'points:',sum(1 for r in positions if r['case_id'] in keep))
    print('exclusion flags:',Counter(flag for r in audit for flag in r['reasons'].split(';') if flag))
    print('retained:',', '.join(sorted(keep)))

if __name__=='__main__': main()

