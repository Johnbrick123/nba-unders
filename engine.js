// The one scoring engine. The board inlines this file; monitor/snapshot.js runs it in Node
// to log each day's pregame rows, so the log always matches what the board showed.
// Weights, fatigue points and projection settings come from model.json (DATA.model).
function makeEngine(DATA){
  const M=DATA.model, P=M.projection, FP=M.fatigue, T=DATA.teams, AB=Object.keys(T), FEAT=DATA.features||{}, INJ=DATA.injuries||{}, PL=DATA.players||{};
  const mean=f=>AB.reduce((s,a)=>s+f(T[a]),0)/AB.length;
  const LG={pts:mean(t=>t.pts),pace:mean(t=>t.pace),ortg:mean(t=>t.ortg),drtg:mean(t=>t.drtg)};
  for(const a of AB){const t=T[a]; t.offR=t.ortg/LG.ortg; t.defR=t.drtg/LG.ortg; t.paceR=t.pace/LG.pace;}
  // League ranks (1 = best offense, 1 = leakiest defense, 1 = fastest) for the top-10 / bottom-10 matchup flags
  const rank=f=>{const s=[...AB].sort((a,b)=>f(T[b])-f(T[a]));const r={};s.forEach((a,i)=>r[a]=i+1);return r};
  const RK={off:rank(t=>t.ortg),def:rank(t=>t.drtg),pace:rank(t=>t.pace)};
  for(const a of AB){T[a].rkOff=RK.off[a];T[a].rkDef=RK.def[a];T[a].rkPace=RK.pace[a];}
  // Matchup label: OVER = top-10 offense vs bottom-10 defense; UNDER = bottom-10 offense vs top-10 defense;
  // STRENGTH = top-10 offense vs top-10 defense; WEAK = bottom-10 offense vs bottom-10 defense. Pace: fast/slow when either side is top/bottom 10.
  function matchup(t,o){
    const top=r=>r<=10, bot=r=>r>=AB.length-9;
    const kind=top(t.rkOff)&&top(o.rkDef)?'OVER':bot(t.rkOff)&&bot(o.rkDef)?'UNDER':top(t.rkOff)&&bot(o.rkDef)?'STRENGTH':bot(t.rkOff)&&top(o.rkDef)?'WEAK':'';
    const pace=top(t.rkPace)||top(o.rkPace)?'fast':bot(t.rkPace)&&bot(o.rkPace)?'slow':'';
    return{kind,pace,label:`O${t.rkOff} vs D${o.rkDef}`};
  }

  // Fatigue index 0-100 from the schedule features (mirrors nbadata.fatigue_points exactly)
  function fatigueIndex(f){
    if(!f)return 0; let p=0;
    if(f.b2b)p+=FP.b2b; if(f.g4>=3)p+=FP.threeInFour; if(f.g7>=4)p+=FP.fourInSeven;
    if(f.trip>=3)p+=FP.tripGame*Math.min(f.trip-2,3);
    p+=FP.travel*Math.min(f.km,3000)/1000; p+=FP.tz*Math.min(f.tz,3);
    if(f.alt)p+=FP.altitude; if(f.rest>=3&&f.ha==='Home')p-=FP.restBonus;
    return Math.max(0,Math.min(100,p));
  }
  // Scoring share of the players a team is missing: Out counts in full, Doubtful 3/4, Questionable 2/5, Day-To-Day 1/4
  const SHARE={'Out':1,'Doubtful':0.75,'Questionable':0.4,'Day-To-Day':0.25};
  function lostShare(ab,exclude){
    const list=INJ[ab]||[], roster=PL[ab]||[], tot=roster.reduce((s,p)=>s+p.pts,0)||1; let lost=0; const who=[];
    for(const i of list){const w=SHARE[i.status]||0; if(!w||(exclude&&exclude[i.pid]))continue;
      const p=roster.find(x=>x.pid===i.pid); const sh=p?p.pts/tot:0;
      lost+=w*sh; who.push({name:i.name,status:i.status,detail:i.detail,ppg:p?p.ppg:null,share:sh,w});}
    return{lost,who:who.sort((a,b)=>b.share-a.share)};
  }

  // 0-100 scales, high = favors scoring for the team in the row
  const span=f=>{const v=AB.map(a=>f(T[a]));return{min:Math.min(...v),max:Math.max(...v)}};
  const SC={off:span(t=>t.ortg),def:span(t=>t.drtg),pace:span(t=>t.pace),fatigue:{min:0,max:60},oppFatigue:{min:0,max:60},
    avail:{min:0,max:0.4},l5:span(t=>t.l5==null?LG.pts:t.l5)};
  const clamp=v=>Math.max(0,Math.min(100,v));
  const lowGood=(v,s)=>v==null?null:clamp((s.max-v)/(s.max-s.min)*100);
  const highGood=(v,s)=>v==null?null:clamp((v-s.min)/(s.max-s.min)*100);

  const Phi=z=>{const t=1/(1+0.2316419*Math.abs(z)),d=0.3989423*Math.exp(-z*z/2),p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))));return z>=0?1-p:p};
  const american=p=>p>=0.5?String(Math.round(-100*p/(1-p))):'+'+Math.round(100*(1-p)/p);
  // Book-implied team totals from the game total and the home spread (negative = home favored)
  const implied=(g,side)=>g.total==null||g.spread==null?null:side==='home'?(g.total-g.spread)/2:(g.total+g.spread)/2;

  function gameTotal(ma,mh){
    const m=2*LG.pts, raw=ma+mh, mu=m+(P.gameShrink??1)*(raw-m);
    return{mu,under:L=>Phi((L-mu)/P.sdGame)};
  }
  function buildRows(date,W,excl){
    W=W||M.weights; excl=excl||{}; const out=[];
    for(const g of DATA.schedule[date]||[])for(const side of ['away','home']){
      const ab=side==='away'?g.away:g.home, opp=side==='away'?g.home:g.away, t=T[ab], o=T[opp];
      if(!t||!o)continue;
      const f=(FEAT[g.id]||{})[ab]||null, of=(FEAT[g.id]||{})[opp]||null;
      const fat=fatigueIndex(f), oppFat=fatigueIndex(of);
      const av=lostShare(ab,excl[ab]), oav=lostShare(opp,excl[opp]);
      const poss=LG.pace*t.paceR*o.paceR;
      const raw=poss/100*LG.ortg*t.offR*o.defR*(side==='home'?P.home:P.away)*(1-P.fatigueK*fat/100)*(1+P.oppFatigueK*oppFat/100)*(1-P.availK*(Math.min(1,av.lost)-P.availBase));
      const mu=LG.pts+P.shrink*(raw-LG.pts);
      const s={off:highGood(t.ortg,SC.off),def:highGood(o.drtg,SC.def),pace:highGood(poss,SC.pace),fatigue:lowGood(fat,SC.fatigue),
        oppFatigue:highGood(oppFat,SC.oppFatigue),avail:lowGood(av.lost,SC.avail),l5:t.l5==null?null:highGood(t.l5,SC.l5)};
      let num=0,den=0,partial=false; for(const k in W){if(s[k]==null){if(W[k]>0)partial=true;continue}num+=s[k]*W[k];den+=W[k]}
      const fin=g.state==='FINAL', actual=side==='away'?g.awayScore:g.homeScore, line=implied(g,side);
      out.push({g,key:`${date}|${g.id}|${ab}`,ab,opp,t,o,ha:side==='home'?'Home':'Away',game:`${g.away} @ ${g.home}`,f,of,fat,oppFat,av,oav,poss,mu,mk:matchup(t,o),
        line,edge:line==null?null:line-mu,pUnder:line==null?null:Phi((line-mu)/P.sdTeam),
        s,total:den?num/den:null,partial,final:fin,actual});
    }
    return out;
  }
  return {M,T,AB,LG,SC,RK,matchup,fatigueIndex,lostShare,lowGood,highGood,Phi,american,implied,buildRows,gameTotal};
}
if(typeof module!=='undefined')module.exports={makeEngine};
