(function(root){
function predict(input,m){
 const x={...input};
 for(const s of m.schema){if(s.type==='number'){const v=Number(x[s.name]);if(x[s.name]===''||!Number.isFinite(v)||v<s.min||v>s.max)throw Error(s.name+' must be between '+s.min+' and '+s.max+'.');x[s.name]=v}else if(!s.options.includes(String(x[s.name])))throw Error('Choose a valid '+s.name+'.')}
 if(x.YearBuilt>Number(x.YrSold)||x.YearRemodAdd>Number(x.YrSold)||x.YearRemodAdd<x.YearBuilt)throw Error('Build/remodel years must be consistent and no later than sale year.');
 if(Math.abs(x.GrLivArea-x['1stFlrSF']-x['2ndFlrSF']-x.LowQualFinSF)>1)throw Error('Living area must equal first floor + second floor + low-quality finished area.');
 if(x.BsmtFinSF1+x.BsmtFinSF2>x.TotalBsmtSF||Math.abs(x.BsmtFinSF1+x.BsmtFinSF2+x.BsmtUnfSF-x.TotalBsmtSF)>1)throw Error('Basement areas must add up to the total basement area.');
 for(const [k,v] of Object.entries({HasPool:x.PoolArea,Has2ndFloor:x['2ndFlrSF'],HasGarage:x.GarageArea,HasBsmt:x.TotalBsmtSF,HasFireplace:x.Fireplaces}))x[k]=v>0?1:0;
 x.TotalSF=x.TotalBsmtSF+x['1stFlrSF']+x['2ndFlrSF'];x.Total_Bathrooms=x.FullBath+.5*x.HalfBath+x.BsmtFullBath+.5*x.BsmtHalfBath;x.Total_porch_sf=x.OpenPorchSF+x['3SsnPorch']+x.EnclosedPorch+x.ScreenPorch;x.YearsSinceRemodel=Number(x.YrSold)-x.YearRemodAdd;
 for(const [k,l] of Object.entries(m.lambdas)){if(x[k]<0)throw Error('Invalid negative value for '+k);x[k]=Math.abs(l)<1e-12?Math.log1p(x[k]):Math.expm1(l*Math.log1p(x[k]))/l;}
 let y=m.intercept;m.mapping.forEach((f,i)=>{y+=m.coefficients[i]*(f.numeric?x[f.numeric]:(String(x[f.category])===f.value?1:0))});const price=Math.expm1(y);if(!Number.isFinite(price)||price<=0)throw Error('These details produce an unsupported estimate. Please review the property inputs.');return price;
}root.predictHouse=predict;if(typeof module!=='undefined')module.exports=predict;
})(globalThis);
