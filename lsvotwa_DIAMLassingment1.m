%No 1
% paper folding formula for alternate-direction
%t*(2)^n > h; % t=thickness of paper n = number of folds and h = height
%(2)^n > h/t;
%n= log2(h/t);
% for it to be more number of folds needed are
n = 1+(log2(8.848/0.001));
disp(n);





%%
%No 2
%initial volume = v(0)
%volume at time t (v(t)) is equal to v(0)exp(-at)
% solving to find time t such that volume = (v(0))/2
%v(0)/2 = v(0)exp(-at);
%1/2 = exp(-at);
%ln(1/2) = ln(exp-at);
%ln(0.5) = -atln(e);
%ln(0.5) = -at;
%t = ln(0.5)/-a;
%a = 0.1;
t = log(0.5)/(-0.1); %ln not a builtin function was replaced by log
disp(t)



%%
%No 3
% A is the initial deposit, r is the rate in percentage terms, n is the number of compounding periods
%formula  fut_val = A(1 + r)^(n); 
%year1 
fut_val1 = 100*(1+0.05)^1;
disp(fut_val1)
%year2
fut_val2 = 100*(1+0.05)^2;
disp(fut_val2)
%year3
fut_val3 = 100*(1+0.05)^3;
disp(fut_val3)
%year4
fut_val4 = 100*(1+0.05)^4;
disp(fut_val4)
%year5 
fut_val5 = 100*(1+0.05)^5;
disp(fut_val5)

%%
%No 4
%formula    r(pv)/1-91+r)^(-n)
%yr1
yr1=(0.01*20000)/(1-(1+0.01)^-12);
disp(yr1)

%yr2
yr2=(0.01*20000)/(1-(1+0.01)^-24);
disp(yr2)

%yr3
yr3=(0.01*20000)/(1-(1+0.01)^-36);
disp(yr3)








%%
%No 5
i = 1;

customers = 100;
payments = 0;
days = 1:9902;
while(tt <= 100000)
   % days=days+1; 
   payments(i) = customers*10;
   customers =  customers+1;
  i = i+1;
   tt = cumsum(payments);
  
  
end

  plot(tt,days)
  xlabel('dates')
  ylabel('cumulative profits')
 title('Cumulative Sum for profits')
 
    hold on
  
  
 



%%
%No 6
T = readtable('ebola_download.xls');
disp(T)
tt = table2timetable(T);
tt(:,:)
(T);
tt2 = retime(tt, 'daily', 'previous');
disp(tt2)

%disp(tt3)
tt3 = timetable2table(tt2);
x = interp1(tt3.Death,tt3.Cases, 'linear');

plot(x)

  xlabel('death')
  ylabel('cases')
 title('graph of cases and death')




%%
%No 7

T = readtable('ebola_download.xls');
disp(T)
tt = table2timetable(T);
tt(:,:)

(T);
tt2 = retime(tt, 'daily', 'previous');
disp(tt2)

tt3 = timetable2table(tt2);

x = interp1(tt3.Death,tt3.Cases, 'linear');
%tt4 = timetable2table(tt3);
tt3(:,2)

z=nanmean(x)*(100);
disp(z)



%%
%No 8
num = xlsread('ebola_download');
disp(num)
T = readtable('ebola_download.xls');
disp(T)
tt = table2timetable(T);
tt(:,:)
(T);
tt2 = retime(tt, 'daily', 'previous');
disp(tt2)

%disp(tt3)
tt3 = timetable2table(tt2);
%V = interp1(tt4.Cases, WeatherData.Death, xq, 'linear');
x = interp1(tt3.Death,tt3.Cases, 'linear');
plot(tt3.Death, tt3.Cases)
 xlabel('Deaths')
  ylabel('Cases')
 title('Plot of death aginst Cases')


%%
%No 9 
T = readtable('SPY.csv');
X = T{:,1};
y = T{:,6};

%normalize 
b=T{:,6}*(100/167.384567);    
disp(b)

T2 = readtable('TLT.csv');
l = T2(:,1);
m = T2{:,6};

%normalise(
m=T2{:,6}*(100/93.096565);
disp(m)

plot(m, T2{:,1})
hold on 
plot(b, T{:,1})
 xlabel('Adjusted Closing Price')
  ylabel('Days')
 title('Plot of adjusted closing prices for 2 time series')
 
 
 %%
 %No 10
 T = readtable('SPY.csv');
% i = 1;
 for t =2:419
     
     r(t) = T{:,6}(t)/T{:,6}(t-1)-1;
 end
  
x=mean(r);
disp(x)
y=min(r);
disp(y)
z=max(r);
disp(z)

 T = readtable('TLT.csv');

for m =2:418
     
     j(m) = T{:,6}(m)/T{:,6}(m-1)-1;
 end
a=mean(j);
disp(a)
b=min(j);
disp(b)
c=max(j);
disp(c)
