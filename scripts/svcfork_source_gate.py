"""Source-keyed soft gate.

The model hums (~560-600 Hz, the interpolated f0 driving the NSF source) in the
gaps where the original singer is silent. Key a gate on the SOURCE vocal's own
envelope: full gain where she sings, ramp toward a floor where she is silent.
Never hard zeros - a 20 dB floor with attack/release keeps the phrase tails and
room that hard gating destroyed earlier.

usage: gate.py <source_vocal> <output> <out.wav>
"""
import sys, numpy as np, soundfile as sf, librosa
from scipy.ndimage import maximum_filter1d
src,outp,dst=sys.argv[1:4]
y,sr=sf.read(outp); stereo=y.ndim>1
s,ssr=sf.read(src); s=(s.mean(1) if s.ndim>1 else s).astype(np.float32)
if ssr!=sr: s=librosa.resample(s,orig_sr=ssr,target_sr=sr)
H=256; N=1024
rms=librosa.feature.rms(y=s,frame_length=N,hop_length=H)[0]
db=20*np.log10(rms+1e-12); db-=db.max()
open_db,close_db,floor_db=-32.0,-46.0,-20.0          # hysteresis window, floor
g=np.clip((db-close_db)/(open_db-close_db),0,1)       # 0 = silent, 1 = singing
g=maximum_filter1d(g,int(0.08*sr/H))                  # 80 ms hold so consonant onsets stay
# attack 15 ms / release 180 ms one-pole smoothing (release follows her tails)
out=np.zeros_like(g); a_up=np.exp(-H/(0.015*sr)); a_dn=np.exp(-H/(0.18*sr)); v=0.0
for i,x in enumerate(g):
    v = x + (v-x)*(a_up if x>v else a_dn); out[i]=v
floor=10**(floor_db/20)
gain=floor+(1-floor)*out
gain=np.interp(np.arange(len(y)), np.arange(len(gain))*H, gain)
yg=(y*gain[:,None]) if stereo else y*gain
sf.write(dst,yg.astype(np.float32),sr)
# report gap level before/after, same definition as gaps.py
m=(y.mean(1) if stereo else y); mg=(yg.mean(1) if stereo else yg)
S=np.abs(librosa.stft(m,n_fft=2048,hop_length=512))**2; Sg=np.abs(librosa.stft(mg,n_fft=2048,hop_length=512))**2
r2=librosa.feature.rms(y=s,frame_length=2048,hop_length=512)[0]; d2=20*np.log10(r2+1e-12); d2-=d2.max()
n=min(S.shape[1],len(d2)); vo=d2[:n]>-30; ga=d2[:n]<-45
lv=10*np.log10(S[:,:n][:,vo].mean()+1e-20)
print(f"  {dst.split('/')[-1]:28} gap-vs-voiced before {10*np.log10(S[:,:n][:,ga].mean()+1e-20)-lv:6.1f} dB -> after {10*np.log10(Sg[:,:n][:,ga].mean()+1e-20)-lv:6.1f} dB   (source: -46.8)")
