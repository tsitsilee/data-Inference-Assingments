%%
%Number 1
load BlueChipStockMoments
assetCovCorr = corrcov(AssetCovar);
[wcoeff,score,latent,tsquared,explained] = pca(assetCovCorr,'VariableWeights','variance');
%[COEFF,latent,explained] = pcacov(assetCovCorr);

%1.3
% plotting for component 1
figure;
CoeffCol1 = wcoeff(:,1)
bar(CoeffCol1)
xlabel('stock');
ylabel('weight');
title('component 1 bar graph');
hold off

%ploting for component 2
figure;
CoeffCol2 = wcoeff(:,2)
bar(CoeffCol2)
xlabel('stock');
ylabel('weight');
title('Component 2 bar graph');
hold off

%1.4
%scree plot of pcacov
figure;
pareto(explained)
title('scree plot');
xlabel('Principal Component')
ylabel('Variance Explained (%)')
hold off

%principle components required
% Number of principal components
prinCompReq = find(cumsum(explained)>= 95, 1);

%testing
ExplainedA = explained(1:12,1);
figure;
pareto(ExplainedA )
title('scree plot');
xlabel('Principal Component')
ylabel('Variance Explained (%)')
hold off

%1.5
%scatter plot 
%calculating mean of all 30 stocks
stocksMeanComp1 = mean(CoeffCol1)
stocksMeanComp2 = mean(CoeffCol2)

%euclidean distance

euclideanDistance = sqrt(((CoeffCol1 -stocksMeanComp1).^2)+ ((CoeffCol2-stocksMeanComp2).^2));

%double checking to see if i get the same answer
pc1EuclidDis = pdist2(wcoeff(:,1), stocksMeanComp1, 'Euclidean');
pc2EuclidDis = pdist2(wcoeff(:,2), stocksMeanComp2, 'Euclidean');

for i = 1:30
    prinCompdistv(i) = sqrt((pc1EuclidDis(i)^2) + (pc2EuclidDis(i)^2));
end
princCompdist = prinCompdistv';
[distance,index] = maxk(prinCompdistv, 3)

%finding 3 max 
%maxDist1 = AssetList{index(1)};
%maxDist2 = AssetList{index(2)};
%maxDist3 = AssetList{index(3)};

coefforth = inv(diag(std(assetCovCorr)))*wcoeff;
c3 = coefforth(:,1:3);

% 2d buiplot
% Visualize the results.
% Visualize both the orthonormal principal component coefficients for each variable and the principal component scores for each observation in a single plot.
figure;
biplot(coefforth(:,1:2),'scores',score(:,1:2),'varlabels',AssetList);
axis([-.26 0.6 -.51 .51]);

hold off
% Check coefficients are orthonormal.
figure;
I = c3'*c3
cscores = zscore(assetCovCorr)*coefforth;
biplot(coefforth(:,1:3),'scores',score(:,1:3),'varlabels',AssetList);
axis([-.26 0.6 -.51 .51]);
view([30 40]);

%finding max points
[st2,index] = sort(tsquared,'descend'); % sort in descending order
extreme = index(1:3);
AssetList(:,extreme)
%%
%Number 2
load BlueChipStockMoments
assetCovCorr = corrcov(AssetCovar);
%[COEFF,latent,explained] = pcacov(assetCovCorr);

%2.3
%Pairwise distances between 30 stocks
pairs=pdist(assetCovCorr);

%formula used
pairwiseDistance = sqrt(2*(1 - assetCovCorr));

%finding large and small dista
largDist=max(pairs)
smallDist=min(pairs)

%2.4
%Horizontal Dendrogram plot
tree = linkage(assetCovCorr,'average');
figure()
horizonatlDendDiag = dendrogram(tree,'Orientation','left','ColorThreshold','default');
set(horizonatlDendDiag,'LineWidth',2);
title('Dendrogram of stocks against pairwise Distance');
xlabel('Pairwise Distance');
ylabel('Stocks');
hold off

%2.5
%Clustering few stocks of the dendrogram
clusteringDen = cluster(tree,'maxclust',3);
figure;
cutoff = median([tree(end-2,3) tree(end-1,3)]);
dendDiag = dendrogram(tree,'Orientation','left','ColorThreshold',cutoff,'Labels',AssetList);
set(dendDiag,'LineWidth',2);
title('Dendrogram of stocks against pairwise distance');
xlabel('Pairwise distance');
ylabel('Stocks');
hold off


%%
%number 3
%num 3.4
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

%categorising data
catClass = categorical(titanictbl1.pclass);
catSex = categorical(titanictbl1.sex);
catAge = categorical(titanictbl1.age);
catSurvival = categorical(titanictbl1.survived);

titatnicCategorical = table(catClass,catSex,catAge, 'VariableNames',{'pclass','sex','age'});
Survival = table(catSurvival,'VariableNames',{'Survived'});

treeModel = TreeBagger(85,titatnicCategorical,Survival,'oobvarimp','on')

% create a visual graphic for the tree
view(treeModel.Trees{1},'mode','graph')

%ploting the graph
figure;
plot(oobError(treeModel));
xlabel('Optimal growth of trees'); 
ylabel('OutOfBag Classification Error');
title('Optimal number Growth of trees against OutOfBag classification Error')
hold off

%Sub3.5
%computing ROC analysis.
idxvar1 = find(treeModel.OOBPermutedVarDeltaError>0.75);
treeb5v = TreeBagger(100,titatnicCategorical(:,idxvar1),Survival,'oobpred','on');
gPosition = find(strcmp('1',treeb5v.ClassNames));


[Yfit,Sfit] = oobPredict(treeb5v);

[fpr,tpr] = perfcurve(treeb5v.Y,Sfit(:,gPosition),'1');
figure;
plot(fpr,tpr, 'r-', 'Linewidth', 1.1)
hold on

%logistic model
logModel = table(catClass,catSex, catAge, catSurvival, 'VariableNames',{'pclass','sex','Age','survived'});
linearModelFit = fitglm(logModel,'Distribution', 'binomial');
 scores = linearModelFit.Fitted.Probability;

 [Xlog,Ylog,T,AUC] = perfcurve(catSurvival,scores,'1');

plot(Xlog,Ylog, 'g-', 'Linewidth', 1.5)

%KNN model
[N,D] = size(titatnicCategorical);

%neighbors
kneighbors = round(logspace(0,log10(N),10)); 
cvloss = zeros(length(kneighbors),1);
for k=1:length(kneighbors)
   
 %cross-validated classification model
  knnMdl = fitcknn(titatnicCategorical,Survival,'NumNeighbors',kneighbors(k));  
  
    %in-sample loss calculation
   rloss1(k)  = resubLoss(knnMdl);
   
  %Cross-validated classifier from KNN model.
 crossMdl = crossval(knnMdl);
    
  %cross-validation loss.
cvloss(k) = kfoldLoss(crossMdl);
 end
 [cvlossmin,icvlossmin] = min(cvloss);
kopt = kneighbors(icvlossmin);
 
knnModel = fitcknn(titatnicCategorical,Survival,'NumNeighbors', kopt, 'Distance', 'euclidean', 'ClassNames',{'1', '0'});
 
%score

[KNNLabel, KNNScore]= resubPredict(knnModel);

Survived = find(strcmp('1',knnModel.ClassNames));  

[FPRknn, TPRknn] = perfcurve(categorical(table2array(Survival)), KNNScore(:,Survived), '1');

plot(FPRknn,TPRknn, 'b-', 'Linewidth', 1.2)

%Classification tree
classTree = fitctree(titatnicCategorical, Survival, 'ClassNames',{'1','0'});
[~,score] = resubPredict(classTree);
 survivedTree = find(strcmp('1',classTree.ClassNames));
[Xtree,Ytree] = perfcurve(categorical(table2array(Survival)),score(:, survivedTree),'1');

plot(Xtree,Ytree, 'k-', 'Linewidth', 1.8)
xlabel('False + Rate')
ylabel('True - Rate')
title('ROC Classification:Logistic Regression, Tree, RF and KNN')
legend('Random Forest','Logistic', 'KNN', 'tree', 'Location', 'best')





%%
%Number 4
redWine = readtable('winequality-red.csv');

names = redWine.Properties. VariableNames;
redWine = table2array(redWine);

predictors = redWine(1:800,1:11);
Quality=redWine(1:800,12);

Predictorstest=redWine(801:end,1:11);
Qualitytest=redWine(801:end,12);

%4.2
mdlTree = TreeBagger(50,predictors,Quality,'oobvarimp','on');
figure
plot(oobError(mdlTree));
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Classification Error');
title('Random forest model for the red wine')

%Estimating the number of leafs
leaf = [1 5 10 20 50 100];
col = 'rgbcmy';

figure
for i=1:length(leaf)
    b = TreeBagger(50,predictors,Quality,'method','r','oobpred','on','minleaf',leaf(i));
    plot(oobError(b),col(i));
    hold on;
end

xlabel('Number of Grown Trees');
ylabel('Mean Squared Error');
legend({'1' '5' '10' '20' '50' '100'});
title('Estimating leaves')
hold off;

%4.3
%Estimating max number of trees
tree = [1 5 10 20 50 100];
col = 'rgbcmy';

figure
for i=1:length(tree)
    b = TreeBagger(tree(i),predictors,Quality,'method','r','oobpred','on','minleaf',1);
    plot(oobError(b));
    hold on;
end
xlabel('Number of Grown Trees');
ylabel('Mean Squared Error');
legend({'1' '5' '10' '20' '50' '100'});
title('max tree number')
hold off;

%4.4
%Feature importance:
mdl1Tree = TreeBagger(50,predictors,Quality,'method','r','oobpred','on','OOBPredictorImportance','on','minleaf',1);

figure
bar(mdl1Tree.OOBPermutedVarDeltaError);
xtickangle(30)
xticklabels({'fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'});
xlabel('Feature Number');
ylabel('Out-Of-Bag Feature Importance');
title('FEATURES IMPORTANCE')


%LASSO COMPARIZON
LASSO=[1 , 2 , 5, 7, 9, 10, 11];
figure
hold on
data=mdlTree.OOBPermutedVarDeltaError;
for i = 1:length(data)
    h=bar(i,data(i),'FaceColor','flat');
    if (any(LASSO(:) == i))
        set(h,'FaceColor','r');
    else
        set(h,'FaceColor','k');
    end
end
xticks([1:11])
xtickangle(30)
xticklabels({'fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'});
xlabel('Feature Index');
ylabel('Out-of-Bag Feature Importance');
title('Importance of feature and compare with LASSO')

% features selected by lasso arw
xVar=find(mdlTree.OOBPermutedVarDeltaError>0.75);
testTree = TreeBagger(100,predictors(:,xVar),Quality,'oobpred','on');
figure
plot(oobError(testTree));
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Classification Error');
title('ROC ANALYSIS')


%4.5
Goodp = find(strcmp('8',testTree.ClassNames));
badp = find(strcmp('3',testTree.ClassNames));

[Yfit,Sfit] = oobPredict(testTree);
[fpr,tpr,T,AUC] = perfcurve(testTree.Y,Sfit(:,Goodp),'8');
figure
plot(fpr,tpr);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Random forest ROC curve');

predictor = predict(mdl1Tree,Predictorstest);

%MSE of RF model
mse = immse(predictor,Qualitytest);
 
%red wine 
variables=redWine(:,1:11);
dependent=redWine(:,12);

%linear regression construction
model= fitlm(variables,dependent);

%optimum K neghbour
indl = [1:size(variables)/2];
indt = [(size(variables)+1)/2:size(variables)];
Xl = variables(indl,:);
yl = dependent(indl,:);
Xt = variables(indt,:);
yt = dependent(indt);
nl = length(yl);
nt = length(yt);

Mean= mean(dependent);

K = 2.^[0:5];
for k=1:length(K)
   [idx, dist] = knnsearch(Xl,Xt,'dist','seuclidean','k',K(k));
   
   ythat = nanmean(yl(idx),2);
   E = yt - ythat;
   MSE(k) = nanmean(E.^2);
   sse(k)=nanmean((yt-Mean).^2);
   sst(k)=nanmean((yt-ythat).^2);
end

rsquareMat=abs(ones(size(sse))-(sse./sst));
u=1:30;
figure;
plot(K,MSE,'k.-');
hold on
plot(K,mse*ones(size(K)))
plot(K,model.MSE*ones(size(K)))

xlabel('Number of nearest neighbors')
ylabel('MSE')
legend('MSE for KNN','MSE for linear Mod','MSE for RF Mod')
title('Model performance')
