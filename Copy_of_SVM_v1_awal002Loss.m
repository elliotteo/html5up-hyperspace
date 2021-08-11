tic
x = squeeze(fullList(:,1,:,:));
y = x(:,1);

x = x(:,[2:135]); % kfoldloss = 0.0014!!!

size(x)
x = squeeze(reshape(x,length(y),1,[]));
size(x)


%generate the test and training sets
%dataset = randsample(length(y),4000);
ytrain = y;
xtrain = x;

% Test dataset
%dataset = randsample(length(y),1200);

disp('finished squeezing the data to fit')
% Train an ECOC multiclass model using the default options.

template = templateSVM('standardize', true,  'KernelFunction' ,'gaussian', 'KernelScale', 25, 'Solver', 'L1QP', 'SaveSupportVectors', 'on', 'BoxConstraint', 1000); % isLoss = 0.000, Kfoldloss = 0.0189 


%Fit model

params = hyperparameters('fitcecoc',xtrain,ytrain,'svm');
params(2).Range = [1e3,1e5]
params(3).Range = [1,1e4]
params = params(3)
Mdl = fitcecoc(xtrain,ytrain, 'Learners',template)

%[Mdl,HyperparameterOptimizationResults] = fitcecoc(xtrain,ytrain, 'Learners',template, 'OptimizeHyperparameters',params,'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',200))


%%
% |Mdl| is a |ClassificationECOC| model.  By default, |fitcecoc| uses SVM
% binary learners, and uses a one-versus-one coding design. You can access
% |Mdl| properties using dot notation.
%%
% Display the coding design matrix.
Mdl.ClassNames
CodingMat = Mdl.CodingMatrix
%% 
% A one-versus-one coding design on three classes yields three binary 
% learners.  Columns of |CodingMat| correspond to learners, and rows 
% correspond to classes.  The class order corresponds to the order
% in |Mdl.ClassNames|. For example, |CodingMat(:,1)| is |[1; -1; 0]| and  
% indicates that the software trains the first SVM binary learner using
%%
% You can access each binary learner using cell indexing and dot notation.
Mdl.BinaryLearners{1}                % The first binary learner
%%
% Compute the in-sample classification error.
isLoss = resubLoss(Mdl)
%%
% The classification error is small, but the classifier might have been
% overfit.  You can cross-validate the classifier using |crossval|.
Cval = crossval(Mdl);
kfoldloss = kfoldLoss(Cval)

 [validationPredictions, validationScores] = kfoldPredict(Cval);
 confmat=confusionmat(xtrain(:,1),validationPredictions);

toc
