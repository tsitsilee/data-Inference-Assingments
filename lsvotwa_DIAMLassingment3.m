%%
%number1

% H0 :? = 7725;
%Ha :?  ? 7725; 
%'Alpha',0.05

%storing the values in an array
womanDailyIntake = [5260, 5470, 5640, 6180, 6390, 6515, 6805, 7515, 7515, 8230, 8770];

%calcualting the mean
xmean = mean(womanDailyIntake);
disp(xmean)

%calculating the standard deviation to double check from the ttest
xstandardDev = std(womanDailyIntake);
disp(xstandardDev)

%calculating ttest
[h,p,ci,stats] = ttest(womanDailyIntake, 7725);
disp(h)
disp(p)
disp(stats)

%calculating the standard Error Mean
stdErrMean = xstandardDev/sqrt(11);
disp(stdErrMean)

%%
%number2
%For Ireland 
% H0 : ? = 74
%Ha:  ?  ? 74
%For Elsewhere 
% H0 : ? = 57
%Ha:  ?  ? 57

mean1 = 74;
mean2 = 57;
stdDev1 = 7.4;
stdDev2 = 7.1;
sampleSize1 = 42;
sampleSize2 = 61;

% calculating the test statistic
testStatistic = (mean1 - mean2)/(sqrt(stdDev1.^2/sampleSize1 + stdDev2.^2/sampleSize2));
disp(testStatistic)

% calculating the degree of freedom
degreeOfFreedom = (stdDev1.^2/sampleSize1 + stdDev2.^2/sampleSize2).^2/(( ((stdDev1.^2/sampleSize1).^2)/(sampleSize1 - 1)) + ((( stdDev2.^2/sampleSize2).^2/(sampleSize2 - 1))) );
disp(degreeOfFreedom)

%calculating pvalue
pValue = 1-tcdf(testStatistic,degreeOfFreedom);
disp(pValue)
%%
%number3
clf;
%reading gdp
gdptable1 = readtable('gdp.csv','ReadVariableNames',true, 'ReadRownames', true);
gdptable1 = removevars(gdptable1, {'IndicatorName','IndicatorCode'});
gdptable2 = rmmissing(gdptable1.x2013);
gdptable3 = gdptable1.x2013;

%reading  Fertility
FertilityRatetable1 = readtable('FertilityRate.csv' ,'ReadVariableNames',true, 'ReadRownames', true);
FertilityRatetable1 = removevars(FertilityRatetable1, {'IndicatorName','IndicatorCode'});
FertilityRatetable2 = rmmissing(FertilityRatetable1.x2013);
FertilityRatetable3 = FertilityRatetable1.x2013;

    scatter(gdptable3, FertilityRatetable3)
  

  xlabel('GDP per capita PP')
  ylabel('Fertility Rate')
 title('scatter plot of Fertility rate versus GDP per capita PP for all countries in 2013')
 hold off
 
 %calculating corellation coeffient
 corellationCoef = corrcoef(gdptable3, FertilityRatetable3, 'Rows','pairwise');
 disp(corellationCoef)
 
 

%%
%number4
%reading the file
clf;
monthlyAverage = readtable('monthly.xls');
monthlyAverage1 = monthlyAverage(1:312,1:2);

%plotting the graph
Figure1 = figure
plot(monthlyAverage1.Var1,monthlyAverage1.AverageHousePrice)
  xlabel('time')
  ylabel('AverageHousePrice')
 title('scatter plot of AverageHousingPrice versus time from Jan 1991 to Dec 2016')
 
% r(t) = [p(t)/p(t-1)]-1
for m = 2:312
monthlyReturns(m) = (monthlyAverage1.AverageHousePrice(m)/monthlyAverage1.AverageHousePrice(m-1))-1;
end

Figure2 = figure
autoCorel = autocorr(monthlyReturns,'NumLags',20,'NumSTD',1.96);
bar(autoCorel)
hold on
%to find CI
%+-1.96/sqrt(312)
yline(0.11 ,'g', 'postive CI');
yline(-0.11, 'r','negative CI' );
  xlabel('lag')
  ylabel('Sample Correlation')
 title('autocorrelation function')
 
 %calculating annualized returns
 AnnualizedReturn = 1;
 for i = 1:312
  AnnualizedReturn = AnnualizedReturn*( 1 + monthlyReturns(i));
 end
  AR = (((AnnualizedReturn)^(1/26))-1)*100;
 disp(AR)


 %%
 %Number5
 clf;
FTse = readtable('FTSE100.csv');
FTse = sortrows(FTse,'Date','ascend');
FTseMon = FTse(:,7);
FTseMon2 = table2array(FTseMon);
%FTSEMon = Ftse2.AdjClose;

monthly = readtable('monthly.xls');
monthTime = monthly(1:312,1:2);

%normalizing data

for m = 2:312
monthly_Returns(m) = (monthTime.AverageHousePrice(m)/monthTime.AverageHousePrice(m-1))-1;
end
monthlyReturnsCum = cumsum(monthly_Returns);
monthlyRetNorm = (monthlyReturnsCum*100)/monthlyReturnsCum(2);

for m = 2:312
FTSE100(m) = (FTseMon2(m)/FTseMon2(m-1))-1;
end
FTSE100Cum = cumsum(FTSE100);
FTSE100Norm = (FTSE100Cum*100)/FTSE100Cum(2);


plot(monthTime.Var1,monthlyRetNorm)
hold on
plot(monthTime.Var1,FTSE100Norm)
 xlabel('Time')
  ylabel('Normalised Monthly Returns')
 title('Plot of cumulative returns')
legend('Monthly Returns', 'FTSE100')
 %calculating annualized Return
 AnnualizedReturn2 = 1;
 for i = 1:312
  AnnualizedReturn2 = AnnualizedReturn2*( 1 + FTSE100(i));
 end
  AR = (((AnnualizedReturn2)^(1/26))-1)*100;
 disp(AR)