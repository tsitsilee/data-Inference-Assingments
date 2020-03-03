%%
%Number 3
diabetestbl = readtable('Diabetes_Data.xlsx');
diabetestarray = diabetestbl{:,1:10};

%correlation matrix of explanatory variables
diabCorrMatr = corrcoef(diabetestarray, 'rows' , 'complete');

%heatmap of the diabCorrMatrf
diabHeat_Map = imagesc(diabCorrMatr);
colorbar('northoutside')
title('heat-map for diabetes explanatory varibales')
xticklabels({'AGE', 'SEX','BMI','BP','S1','S2','S3','S4','S5','S6'})
yticklabels({'AGE', 'SEX','BMI','BP','S1','S2','S3','S4','S5','S6'})

%Building multivarient model
Y = diabetestbl{:,11};
%[b,bint,r,rint,stats] = regress(Y,diabetestarray);
fitlm(diabetestarray,Y,'Linear')

%doing stepwise for the diabetesarray
diabStepWise = stepwise(diabetestarray,Y);



%%
%Question 4
titanictbl = readtable('titanic3.csv');

%probability of survival of a passenger
titanicSurvived = titanictbl(strcmp(num2str(titanictbl.survived),"1"),:);
srvivalProbability = height(titanicSurvived)/height(titanictbl);
disp(" Probability of survival for a passenger: "+srvivalProbability);

%probability of male survivors
male = titanicSurvived.sex(titanicSurvived.sex=="male");
male2 = height(cell2table(male))
srvivdMen = male2/length(titanictbl.sex=="male");
disp(" Probability of male survivors for the male sample: "+srvivdMen)

%probability of female survivors
female = titanicSurvived.sex(titanicSurvived.sex=="female");
female2 = height(cell2table(female));
srvivdWomen = female2/length(titanictbl.sex=="female");
disp(" Probability of female survivors for the female sample: "+srvivdWomen)

%probability of class1 survivors
class1 = titanicSurvived.pclass(titanicSurvived.pclass==1);
class1tbl = height(array2table(class1));
srvivdClass1 = class1tbl/length(titanictbl.pclass==1);
disp(" Probability of class 1 survivors for class 1 sample: "+srvivdClass1)

%probability of class2 survivors
class2 = titanicSurvived.pclass(titanicSurvived.pclass==2);
class2tbl = height(array2table(class2));
srvivdClass2 = class2tbl/length(titanictbl.pclass==2);
%height(titanicSurvived);
disp(" Probability of class 2 survivors for class 2 sample: "+srvivdClass2)

%probability of class3 survivors
class3 = titanicSurvived.pclass(titanicSurvived.pclass==3);
class3tbl = height(array2table(class3));
srvivdClass3 = class3tbl/length(titanictbl.pclass==3);
disp(" Probability of class 3 survivors for class 3 sample: "+srvivdClass3)

%using a range off 16 per age group
%propbability of ages 0-15 survival
ageGroup0to15 = find(titanicSurvived.age>=0 & titanicSurvived.age<=15);
ageGroup0to15S = height(array2table(ageGroup0to15));
srvivdAges0t15 = ageGroup0to15S/length(find(titanictbl.age>=0 & titanictbl.age<=15));
disp(" Probability of ages 0-15 survivors for that age group: "+srvivdAges0t15)

%propbability of ages 16-32 survival
ageGroup16to32 = find(titanicSurvived.age>=16 & titanicSurvived.age<=32);
ageGroup16to32S = height(array2table(ageGroup16to32));
srvivdAges16t32 = ageGroup16to32S/length(find(titanictbl.age>=16 & titanictbl.age<=32));
disp(" Probability of ages 16-32 survivors for that age group: "+srvivdAges16t32)

%propbability of ages 33-49 survival
ageGroup33to49 = find(titanicSurvived.age>=33 & titanicSurvived.age<=49);
ageGroup33to49S = height(array2table(ageGroup33to49));
srvivdAges33t49 = ageGroup33to49S/length(find(titanictbl.age>=33 & titanictbl.age<=49));
disp(" Probability of ages 33-49 survivors for that age group: "+srvivdAges33t49)

%propbability of ages 50-65 survival
ageGroup50to65 = find(titanicSurvived.age>=50 & titanicSurvived.age<=65);
ageGroup50to65S = height(array2table(ageGroup50to65));
srvivdAges50t65 = ageGroup50to65S/length(find(titanictbl.age>=50 & titanictbl.age<=65));
disp(" Probability of ages 49-65 survivors for that age group: "+srvivdAges50t65)

%propbability of ages 66-81 survival
ageGroup66to81 = find(titanicSurvived.age>=66 & titanicSurvived.age<=81);
ageGroup66to81S = height(array2table(ageGroup66to81));
srvivdAges66t81 = ageGroup66to81S/length(find(titanictbl.age>=66 & titanictbl.age<=81));
disp(" Probability of ages 66-81 survivors for that age group: "+srvivdAges66t81)

%creating the table of all probabilities
probtable = table(categorical({'SurvivedMen';'SurvivedWomen';'SurvivedClass1';'SurvivedClass2';'SurvivedClass3';'Survivedages0to15';'SurvivedAges16to32';'SurvivedAges33to49';'SurvivedAges50to65';'survivedAges66to81'}),...
    [srvivdMen;srvivdWomen;srvivdClass1;srvivdClass2;srvivdClass3;srvivdAges0t15;srvivdAges16t32;srvivdAges33t49;srvivdAges50t65;srvivdAges66t81],'VariableNames',{'Predicted_Variables','Probability'});
disp(probtable)

%building a logistic regresion model

titanictbl2=titanictbl(:,[1,4,5,2]);

titanicNew = 1:1309;
maleInd = find(ismember(titanictbl.sex,"male"));
titanicNew(maleInd) = 0;
FemaleInd = find(ismember(titanictbl.sex,"female"));
titanicNew(FemaleInd) = 1;
titanictbl2.sex=titanicNew';

rModel = fitglm(titanictbl2,'distribution','binomial');
predictModel = predict(rModel,titanictbl2);
newPredictValues = round(predictModel)
confussionMatrix = confusionmat(titanictbl2.survived,newPredictValues)

