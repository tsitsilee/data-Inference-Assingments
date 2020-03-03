%%
%number 2

% acknowledgement tutorial 10a
titanictbl = readtable('titanic3.csv');
titanictbl1 = titanictbl(:,[1:2,4:5]);

%changing age to enable it to be easily categorized
titanicNew = 1:1309;
maleInd = find(ismember(titanictbl.sex,"male"));
titanicNew(maleInd) = 0;
FemaleInd = find(ismember(titanictbl.sex,"female"));
titanicNew(FemaleInd) = 1;
titanictbl1.sex=titanicNew';

titanicNew2 = 1:1309;
ageGroup0to20 = find(titanictbl1.age>=0 & titanictbl1.age<=20);
titanicNew2(ageGroup0to20) = 0;

ageGroup21to40 = find(titanictbl1.age>=21 & titanictbl1.age<=40);
titanicNew2(ageGroup21to40) = 1;

ageGroup41to60 = find(titanictbl1.age>=41 & titanictbl1.age<=60);
titanicNew2(ageGroup41to60) = 2;

ageGroup61to80 = find(titanictbl1.age>=61 & titanictbl1.age<=80);
titanicNew2(ageGroup61to80) = 3;

ageGroupisNan = find(isnan(titanictbl1.age) == 1);
titanicNew2(ageGroupisNan) =NaN ;
titanictbl1.age=titanicNew2';

%categorising data
catClass = categorical(titanictbl1.pclass);
catSex = categorical(titanictbl1.sex);
catAge = categorical(titanictbl1.age);
catSurvival = categorical(titanictbl1.survived);
titatnicCategorical = table(catClass,catSex,catAge, 'VariableNames',{'pclass','sex','age'});


%spliting data 80:20
survivedVariable = table(catSurvival, 'VariableNames',{'survived'});
survivedVariableTraining = survivedVariable(1:1047,1);
survivedVariableTest = survivedVariable(1048:end,1);
categorcalSurvived = categorical(table2array(survivedVariableTest));
titatnicCategoricalTrainingSet = titatnicCategorical([1:1047],[1:3]);
titatnicCategoricalTestSet = titatnicCategorical([1048:1309],[1:3]);

% Classification Tree
classificationtree = ClassificationTree.fit(titatnicCategoricalTrainingSet,survivedVariableTraining)

% create a visual graphic for the tree
view(classificationtree,'mode','graph')

% in-sample evaluation of classificationtree
resuberrorClassTree = resubLoss(classificationtree)

% cross-validation evaluation
cvctreeClassTree = crossval(classificationtree);
cvlossClassTree = kfoldLoss(cvctreeClassTree)

[~,~,~,bestlevel] = cvLoss(classificationtree,'subtrees','all','treesize','min')

% Prune the tree to use it for other purposes:
prunnedtree = prune(classificationtree,'Level',bestlevel);
view(prunnedtree,'mode','graph')

resuberrorPrunnedTree = resubLoss(prunnedtree)

 %cross-validation evaluation
cvcPrunnedTree = crossval(prunnedtree);
cvlossPrunnedTree = kfoldLoss(cvcPrunnedTree)

%prediction for Classiification and Prunned Tree
predctClassTree = predict(classificationtree,titatnicCategoricalTestSet);
predctPrundTree = predict(prunnedtree,titatnicCategoricalTestSet);

%calculating accuracy for pruned tree
ConfusionMatrixPruned = confusionmat(categorcalSurvived,predctPrundTree)
accuracyPrunedTree = (ConfusionMatrixPruned(1,1) + ConfusionMatrixPruned(2,2))/(ConfusionMatrixPruned(1,1)+ConfusionMatrixPruned(1,2)+ConfusionMatrixPruned(2,1)+ConfusionMatrixPruned(2,2))

%calculating accuracy for unprunned tree
ConfusionMatrixUnpruned = confusionmat(categorcalSurvived,predctClassTree)
accuracyUnPrunedTree = (ConfusionMatrixUnpruned(1,1) + ConfusionMatrixUnpruned(2,2))/(ConfusionMatrixUnpruned(1,1)+ConfusionMatrixUnpruned(1,2)+ConfusionMatrixUnpruned(2,1)+ConfusionMatrixUnpruned(2,2))

%logistic regression 
titatnicCategorical2 = table(catClass,catSex,catAge,catSurvival, 'VariableNames',{'pclass','sex','age','survived'});
survivedVar2 = table(catSurvival, 'VariableNames',{'survived'});
survivedVariableTraining2 = survivedVar2(1:1047,1);
survivedVariableTest2 = survivedVar2(1048:end,1);
categorcalSurvived2 = categorical(table2array(survivedVariableTest2));
titatnicCategoricalTrainingSet2 = titatnicCategorical2([1:1047],[1:4]);
titatnicCategoricalTestSet2 = titatnicCategorical2([1048:1309],[1:4]);
rModel = fitglm(titatnicCategoricalTrainingSet2,'distribution','binomial');

%predicting for LogReg
LogpredictModel = predict(rModel,titatnicCategoricalTestSet2);
newPredictValues2 = round(LogpredictModel)
newPredictValues3 = categorical(newPredictValues2)

%calculating accuracy for Log Reg
ConfusionMatrixLogistic = confusionmat(categorcalSurvived,newPredictValues3)
accuracylogReg = (ConfusionMatrixLogistic(1,1) + ConfusionMatrixLogistic(2,2))/(ConfusionMatrixLogistic(1,1)+ConfusionMatrixLogistic(1,2)+ConfusionMatrixLogistic(2,1)+ConfusionMatrixLogistic(2,2))


%%
%Number 3

%acknowledgement tutorial 10b

% Construct the classifier using ClassificationKNN.fit.
titanictbl = readtable('titanic3.csv');
titanictbl1 = titanictbl(:,[1:2,4:5]);

%changing sex to numerical
titanicNew = 1:1309;
maleInd = find(ismember(titanictbl.sex,"male"));
titanicNew(maleInd) = 0;
FemaleInd = find(ismember(titanictbl.sex,"female"));
titanicNew(FemaleInd) = 1;
titanictbl1.sex=titanicNew';

%changing age group to ranges so as to stream it down
titanicNew2 = 1:1309;
ageGroup0to20 = find(titanictbl1.age>=0 & titanictbl1.age<=20);
titanicNew2(ageGroup0to20) = 0;

ageGroup21to40 = find(titanictbl1.age>=21 & titanictbl1.age<=40);
titanicNew2(ageGroup21to40) = 1;

ageGroup41to60 = find(titanictbl1.age>=41 & titanictbl1.age<=60);
titanicNew2(ageGroup41to60) = 2;

ageGroup61to80 = find(titanictbl1.age>=61 & titanictbl1.age<=80);
titanicNew2(ageGroup61to80) = 3;

ageGroupisNan = find(isnan(titanictbl1.age) == 1);
titanicNew2(ageGroupisNan) =NaN ;
titanictbl1.age=titanicNew2';

%categorising data
catClass = categorical(titanictbl1.pclass);
catSex = categorical(titanictbl1.sex);
catAge = categorical(titanictbl1.age);
catSurvival = categorical(titanictbl1.survived);
titatnicCategorical = table(catClass,catSex,catAge, 'VariableNames',{'pclass','sex','age'});
Survival = table(catSurvival,'VariableNames',{'Survived'});

knnmdl = ClassificationKNN.fit(titatnicCategorical,Survival)


% Examine the resubstitution loss, which, by default, is the fraction of misclassifications from the predictions of mdl. (For nondefault cost, weights, or priors, see ClassificationKNN.loss.)
rlossknn = resubLoss(knnmdl)

% Construct a cross-validated classifier from the knnmodel.
cvknnmdl = crossval(knnmdl);
klossknn = kfoldLoss(cvknnmdl)


% KNN classification error versus number of neighbors
[N,D] = size(titatnicCategorical);
K = round(logspace(0,log10(N),10)); % number of neighbors
cvloss = zeros(length(K),1);
for k=1:length(K)
    % Construct a cross-validated classification model
    knnmdl1 = ClassificationKNN.fit(titatnicCategorical,Survival,'NumNeighbors',K(k));
    % Calculate the in-sample loss
    rloss(k)  = resubLoss(knnmdl1);
    % Construct a cross-validated classifier from the model.
    cvmdl = crossval(knnmdl1);
    % Examine the cross-validation loss, which is the average loss of each cross-validation model when predicting on data that is not used for training.
    cvloss(k) = kfoldLoss(cvmdl);
end
[cvlossmin,icvlossmin] = min(cvloss);
kopt = K(icvlossmin);


% plot the accuracy versus k
figure; 
semilogx(K,rloss,'g.-');
hold
semilogx(K,cvloss,'b.-');
plot(K(icvlossmin),cvloss(icvlossmin),'ro')
xlabel('Number of nearest neighbors');
ylabel('Ten-fold classification error');
legend('In-sample','Out-of-sample','Optimum','Location','NorthWest')
title('KNN classification');

knnmd2 = ClassificationKNN.fit(titatnicCategorical,Survival,'NumNeighbors',kopt);

%logistic regression 
titatnicCategorical2 = table(catClass,catSex,catAge,catSurvival, 'VariableNames',{'pclass','sex','age','survived'});
survivedVar2 = table(catSurvival, 'VariableNames',{'survived'});
catSurvived3 = categorical(table2array(Survival));

categorcalSurvived2 = categorical(table2array(survivedVar2));
rModel = fitglm(titatnicCategorical2,'distribution','binomial');
survivedVariable3 = categorical(table2array(survivedVar2));

%predicting for LogReg
LogpredictModel = predict(rModel,titatnicCategorical2);
newPredictValues2 = round(LogpredictModel);
newPredictValues3 = categorical(newPredictValues2);

%calculating accuracy for Log Reg
ConfusionMatrixLogistic = confusionmat(categorcalSurvived2,newPredictValues3);
accuracylogReg = (ConfusionMatrixLogistic(1,1) + ConfusionMatrixLogistic(2,2))/(ConfusionMatrixLogistic(1,1)+ConfusionMatrixLogistic(1,2)+ConfusionMatrixLogistic(2,1)+ConfusionMatrixLogistic(2,2))


modelspec = 'survived ~ pclass + sex + age';

%knn
predKnn = predict(knnmd2,titatnicCategorical); 
newknnPredictValues3 = categorical(predKnn);

%calculating accuracy for knn
knnConfusionMat = confusionmat(survivedVariable3,newknnPredictValues3);
accuracyknn = (knnConfusionMat(1,1) + knnConfusionMat(2,2))/(knnConfusionMat(1,1)+knnConfusionMat(1,2)+knnConfusionMat(2,1)+knnConfusionMat(2,2))

%evaluating performance with different distance matric
%haming
modelhaming = fitcknn(titatnicCategorical,Survival,'Distance','hamming');
rlosshaming = resubLoss(modelhaming)
crossValidationhaming = crossval(modelhaming);
crossValidationLosshaming = kfoldLoss(crossValidationhaming)

%minkowski
modelminkowski = fitcknn(titatnicCategorical,Survival,'Distance','minkowski');
rlossminkowski = resubLoss(modelminkowski)
crossValidationModelminkowski= crossval(modelminkowski);
crossValidationLossminkowski = kfoldLoss(crossValidationModelminkowski)

%euclidean
modeleuclidean = fitcknn(titatnicCategorical,Survival,'Distance','euclidean');
rlossSeuclidean = resubLoss(modeleuclidean)
crossValidationModeleuclidean = crossval(modeleuclidean);
crossValidationLosseuclidean = kfoldLoss(crossValidationModeleuclidean)
%%
%Number 4
redWinetbl = readtable('winequality-red.csv');
whiteWinetbl = readtable('winequality-white.csv');

%calculating averages for each attributes for the red wine
avgfixedAccidityRW = mean(redWinetbl.fixedAcidity)
avgvolatileAcidRW = mean(redWinetbl.volatileAcidity)
avgcitricAcidRW = mean(redWinetbl.citricAcid)
avgresSugarRW = mean(redWinetbl.residualSugar)
avgchloridesRW = mean(redWinetbl.chlorides)
avgfreeSulfurDioxideRW = mean(redWinetbl.freeSulfurDioxide)
avgttlSulfurDioxideRW = mean(redWinetbl.totalSulfurDioxide)
avgdensityRW = mean(redWinetbl.density)
avgphRW = mean(redWinetbl.pH)
avgsulphatesRW = mean(redWinetbl.sulphates)
avgalcoholRW = mean(redWinetbl.alcohol)
avgqualityRW = mean(redWinetbl.quality)

%calculating averages for each attributes for the white wine
avgfixedAccidityWW = mean(whiteWinetbl.fixedAcidity)
avgvolatileAcidWW = mean(whiteWinetbl.volatileAcidity)
avgcitricAcidWW = mean(whiteWinetbl.citricAcid)
avgresSugarWW = mean(whiteWinetbl.residualSugar)
avgchloridesWW = mean(whiteWinetbl.chlorides)
avgfreeSulfurDioxideWW = mean(whiteWinetbl.freeSulfurDioxide)
avgttlSulfurDioxideWW = mean(whiteWinetbl.totalSulfurDioxide)
avgdensityWW = mean(whiteWinetbl.density)
avgphWW = mean(whiteWinetbl.pH)
avgsulphatesWW = mean(whiteWinetbl.sulphates)
avgalcoholWW = mean(whiteWinetbl.alcohol)
avgqualityWW = mean(whiteWinetbl.quality)

%plotting bar graph
c = categorical({'fixeAcidity','volatileAcid','citricAcid','residualSugar','chlroides','freeSO2','ttlSO2','density','ph','sulphates','alcohol', 'quality'});
rednwhiteWine = [avgfixedAccidityRW avgfixedAccidityWW; avgvolatileAcidRW avgvolatileAcidWW;...
    avgcitricAcidRW avgcitricAcidWW; avgresSugarRW avgresSugarWW; avgchloridesRW avgchloridesWW;...
    avgfreeSulfurDioxideRW avgfreeSulfurDioxideWW; avgttlSulfurDioxideRW avgttlSulfurDioxideWW;...
    avgdensityRW avgdensityWW; avgphRW avgphWW; avgsulphatesRW avgsulphatesWW; avgalcoholRW avgalcoholWW;...
    avgqualityRW avgqualityWW];

Figure1 = figure
bar(c,rednwhiteWine)
 xlabel('wine Features')
 ylabel('average values')
 title('bar graph plot comparing red against white wine')
hold off
legend('RedWine', 'WhiteWine')

%correlation matrix of redwine variables
RedWinMatr = corrcoef(table2array(redWinetbl), 'rows' , 'complete');

Figure2 = figure
varNames = {'fxcidity','volAcdty','C6H8O7',...
     'rSgar','Cl-','FrSO2',...
     'totlSO?','dnsty','pH','SO?²-','alcohl','quality'};
 %heatmap of wine-quality-red
 redWinHeat_Map = heatmap( varNames, varNames, RedWinMatr);
 title('wine-quality-red variables correlation');
 xlabel('features');
 ylabel('features');
 colorbar ;
 
%correlation matrix of whitewine variables
whiteWinMatr = corrcoef(table2array(whiteWinetbl), 'rows' , 'complete');

Figure3 = figure
%heatmap of the diabCorrMatr
 whiteWinHeat_Map = heatmap( varNames, varNames, whiteWinMatr);
 title('wine-quality-white variables correlation');
 xlabel('features');
 ylabel('features');
 colorbar ;
 
 %lasso
 redWineArray = table2array(redWinetbl);
 whiteWineArray = table2array(whiteWinetbl);
 
lassoRedX = redWineArray(:,1:11);
lassoWhiteX = whiteWineArray(:,1:11);

lassoRedY = redWinetbl(:,12);
lassoWhiteY = whiteWinetbl(:,12);

xRed = redWineArray(:,1:11);
xWhite = whiteWineArray(:,1:11);

yRed = redWineArray(:,12);
yWhite = whiteWineArray(:,12);

% Lasso  for RED WINE
%cross-validated lasso regularization of a linear regression model
[B,FitInfoRedWine] = lasso(xRed,yRed,'CV',10);

% cross-validation plot .
%Figure4 = figure
lassoPlot(B,FitInfoRedWine,'plottype','CV');

% parameter estimates plot as a function of the Lambda regularization parameter.
%Figure5 = figure
lassoPlot(B,FitInfoRedWine,'PlotType','Lambda','XScale','log');
title('crossvalidated MSE & Lasso for RedWine');

% Find the nonzero model coefficients corresponding to the two identified points.
minpts = find(B(:,FitInfoRedWine.IndexMinMSE));
min1pts = find(B(:,FitInfoRedWine.Index1SE));

FitInfoRedWine.Intercept(FitInfoRedWine.Index1SE);

% Lasso  for WHITE WINE
% cross-validated lasso regularization of the linear regression model.
[BW,FitInfoWhiteWine] = lasso(xWhite,yWhite,'CV',10);

% Examine the cross-validation plot.
%Figure6 = figure
lassoPlot(BW,FitInfoWhiteWine,'plottype','CV');

% plot parameter estimates as function of Lambda regularization parameter.
%Figure7 = figure
lassoPlot(BW,FitInfoWhiteWine,'PlotType','Lambda','XScale','log');
title('crossValidated MSE & Lasso for whiteWine');

% KNN Regression 
xRK = redWineArray(:,[2,4,5,6,7,9,10,11]);  

% learnig and testing datasets
A = length(xRK);
B = round(A*.7);
indl = [1:B];
indt = [B+1:A];
Xl = xRK(indl,:);
yl = yRed(indl,:);
Xt = xRK(indt,:);
yt = yRed(indt);
nl = length(yl);
nt = length(yt);


K = 2.^[0:5];
for k=1:length(K)
   [idx, dist] = knnsearch(Xl,Xt,'dist','seuclidean','k',K(k));
   [idx1, dist1] = knnsearch(Xl,Xt,'dist','mahalanobis','k',K(k));
   ythat = nanmean(yl(idx),2);
   E = yt - ythat;
   RMSE(k) = sqrt(nanmean(E.^2));
   MSE(k) = nanmean(E.^2);
end
  
 RsquaredKNN = sum((ythat-(mean(yt))).^2) / sum((yt-(mean(yt))).^2);
 mseKNN = MSE(k);

Figure8 = figure
plot(K,RMSE,'k.-');
xlabel('Nearest neighbors Number')
ylabel('RMSE')

OLS  = fitlm(xRK, yRed);
olsMSE  = OLS.MSE;
olsRsquared = OLS.Rsquared.Ordinary;

Figure9 = figure
barh([OLS.MSE MSE(k)])
set(gca,'YTick',[1 2]);
set(gca,'YTickLabel',{'OLS MSE','KNN MSE'});


Figure10 = figure
barh([0.41, 0.44, 0.42])
set(gca,'YTick',[1 2 3]);
set(gca,'YTickLabel',{'OLS MSE','KNN MSE', 'RF MSE'});