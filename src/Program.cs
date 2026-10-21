using System;

namespace CodingBackProp
{
  class Program
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\n=======================================================");
      Console.WriteLine("  Neural Network Back-Propagation Demo (CS345 IS 1)");
      Console.WriteLine("=======================================================");

      int numInput = 4;
      int numHidden = 5;
      int numOutput = 3;
      int numRows = 1000;
      int seed = 1;

      Console.WriteLine($"\nGenerating {numRows} artificial data items with {numInput} features");
      double[][] allData = DataUtils.MakeAllData(numInput, numHidden, numOutput, numRows, seed);
      Console.WriteLine("Done");

      Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
      DataUtils.SplitTrainTest(allData, 0.80, seed, out double[][] trainData, out double[][] testData);
      Console.WriteLine("Done\n");

      Console.WriteLine("Training data:");
      DataUtils.ShowMatrix(trainData, 4, 2, true);
      Console.WriteLine("Test data:");
      DataUtils.ShowMatrix(testData, 4, 2, true);

      int maxEpochs = 1000;
      double learnRate = 0.05;
      double momentum = 0.01;

      // -----------------------------------------------------------------
      // Experiment 1 — Baseline: MSE loss, no regularization
      // -----------------------------------------------------------------
      Console.WriteLine("\n=======================================================");
      Console.WriteLine(" EXPERIMENT 1: Baseline (Mean Squared Error, No Decay)");
      Console.WriteLine("=======================================================");
      Console.WriteLine($"Creating a {numInput}-{numHidden}-{numOutput} neural network");
      var nn1 = new NeuralNetwork(numInput, numHidden, numOutput);

      Console.WriteLine($"\nSetting maxEpochs = {maxEpochs}");
      Console.WriteLine($"Setting learnRate = {learnRate:F2}");
      Console.WriteLine($"Setting momentum  = {momentum:F2}");
      Console.WriteLine("Setting errorType = Mean Squared Error");
      Console.WriteLine("Setting decayRate = 0.00 (No Weight Decay)");

      Console.WriteLine("\nStarting training");
      double[] weights1 = nn1.Train(trainData, maxEpochs, learnRate, momentum,
        ErrorType.MeanSquared, 0.0);
      Console.WriteLine("Done");
      Console.WriteLine("\nFinal neural network model weights and biases:\n");
      DataUtils.ShowVector(weights1, 2, 10, true);

      double trainErr1   = nn1.MeanSquaredError(trainData);
      double trainAcc1   = nn1.Accuracy(trainData);
      double testAcc1    = nn1.Accuracy(testData);
      double weightNorm1 = nn1.WeightNorm();
      Console.WriteLine($"\nFinal MSE on training data      = {trainErr1:F4}");
      Console.WriteLine($"Final weight norm (L2 norm)     = {weightNorm1:F4}");
      Console.WriteLine($"Final accuracy on training data = {trainAcc1:F4}");
      Console.WriteLine($"Final accuracy on test data     = {testAcc1:F4}");

      // -----------------------------------------------------------------
      // Experiment 2 — Cross-Entropy loss, no regularization
      // -----------------------------------------------------------------
      Console.WriteLine("\n=======================================================");
      Console.WriteLine(" EXPERIMENT 2: Cross-Entropy Error (No Weight Decay)");
      Console.WriteLine("=======================================================");
      Console.WriteLine($"Creating a {numInput}-{numHidden}-{numOutput} neural network");
      var nn2 = new NeuralNetwork(numInput, numHidden, numOutput);

      Console.WriteLine($"\nSetting maxEpochs = {maxEpochs}");
      Console.WriteLine($"Setting learnRate = {learnRate:F2}");
      Console.WriteLine($"Setting momentum  = {momentum:F2}");
      Console.WriteLine("Setting errorType = Cross-Entropy Error");
      Console.WriteLine("Setting decayRate = 0.00 (No Weight Decay)");

      Console.WriteLine("\nStarting training");
      double[] weights2 = nn2.Train(trainData, maxEpochs, learnRate, momentum,
        ErrorType.CrossEntropy, 0.0);
      Console.WriteLine("Done");
      Console.WriteLine("\nFinal neural network model weights and biases:\n");
      DataUtils.ShowVector(weights2, 2, 10, true);

      double trainErr2   = nn2.CrossEntropyError(trainData);
      double trainAcc2   = nn2.Accuracy(trainData);
      double testAcc2    = nn2.Accuracy(testData);
      double weightNorm2 = nn2.WeightNorm();
      Console.WriteLine($"\nFinal Cross-Entropy on train    = {trainErr2:F4}");
      Console.WriteLine($"Final weight norm (L2 norm)     = {weightNorm2:F4}");
      Console.WriteLine($"Final accuracy on training data = {trainAcc2:F4}");
      Console.WriteLine($"Final accuracy on test data     = {testAcc2:F4}");

      // -----------------------------------------------------------------
      // Experiment 3 — Cross-Entropy loss with L2 weight decay
      // -----------------------------------------------------------------
      double decayRate = 0.001;
      Console.WriteLine("\n=======================================================");
      Console.WriteLine(" EXPERIMENT 3: Cross-Entropy WITH Weight Decay (Regularization)");
      Console.WriteLine("=======================================================");
      Console.WriteLine($"Creating a {numInput}-{numHidden}-{numOutput} neural network");
      var nn3 = new NeuralNetwork(numInput, numHidden, numOutput);

      Console.WriteLine($"\nSetting maxEpochs = {maxEpochs}");
      Console.WriteLine($"Setting learnRate = {learnRate:F2}");
      Console.WriteLine($"Setting momentum  = {momentum:F2}");
      Console.WriteLine("Setting errorType = Cross-Entropy Error");
      Console.WriteLine($"Setting decayRate = {decayRate:F4} (L2 Regularization)");

      Console.WriteLine("\nStarting training");
      double[] weights3 = nn3.Train(trainData, maxEpochs, learnRate, momentum,
        ErrorType.CrossEntropy, decayRate);
      Console.WriteLine("Done");
      Console.WriteLine("\nFinal neural network model weights and biases:\n");
      DataUtils.ShowVector(weights3, 2, 10, true);

      double trainErr3   = nn3.CrossEntropyError(trainData);
      double trainAcc3   = nn3.Accuracy(trainData);
      double testAcc3    = nn3.Accuracy(testData);
      double weightNorm3 = nn3.WeightNorm();
      Console.WriteLine($"\nFinal Cross-Entropy on train    = {trainErr3:F4}");
      Console.WriteLine($"Final weight norm (L2 norm)     = {weightNorm3:F4}");
      Console.WriteLine($"Final accuracy on training data = {trainAcc3:F4}");
      Console.WriteLine($"Final accuracy on test data     = {testAcc3:F4}");

      // -----------------------------------------------------------------
      // Summary table
      // -----------------------------------------------------------------
      Console.WriteLine("\n==========================================================================");
      Console.WriteLine(" COMPARISON OF BACK-PROPAGATION CONFIGURATIONS");
      Console.WriteLine("==========================================================================");
      Console.WriteLine($"{"Configuration",-32} {"Train Loss",-12} {"Weight Norm",-14} {"Train Acc",-12} {"Test Acc",-12}");
      Console.WriteLine(new string('-', 84));
      Console.WriteLine($"{"1. Baseline (MSE, No Decay)",-32} {trainErr1,-12:F4} {weightNorm1,-14:F4} {trainAcc1,-12:P2} {testAcc1,-12:P2}");
      Console.WriteLine($"{"2. Cross-Entropy (No Decay)",-32} {trainErr2,-12:F4} {weightNorm2,-14:F4} {trainAcc2,-12:P2} {testAcc2,-12:P2}");
      Console.WriteLine($"{"3. Cross-Entropy + Weight Decay",-32} {trainErr3,-12:F4} {weightNorm3,-14:F4} {trainAcc3,-12:P2} {testAcc3,-12:P2}");
      Console.WriteLine(new string('-', 84));
      Console.WriteLine("\nKey Observations:");
      Console.WriteLine("  1. Cross-Entropy avoids the softmax derivative saturation term (1-y)*y,");
      Console.WriteLine("     yielding faster learning signals for classification.");
      Console.WriteLine("  2. Weight Decay penalizes large weights, significantly reducing the");
      Console.WriteLine($"     L2 weight norm ({weightNorm2:F2} -> {weightNorm3:F2}), helping prevent over-fitting.");
      Console.WriteLine("==========================================================================\n");

      Console.WriteLine("End back-propagation demo\n");
      Console.ReadLine();
    }
  }
}
