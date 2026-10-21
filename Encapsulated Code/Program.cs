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

      int numInput = 4;  // number features
      int numHidden = 5;
      int numOutput = 3; // number of classes for Y
      int numRows = 1000;
      int seed = 1;      // reproducible demo seed

      Console.WriteLine("\nGenerating " + numRows +
        " artificial data items with " + numInput + " features");
      double[][] allData = DataUtils.MakeAllData(numInput, numHidden, numOutput, numRows, seed);
      Console.WriteLine("Done");

      Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
      double[][] trainData;
      double[][] testData;
      DataUtils.SplitTrainTest(allData, 0.80, seed, out trainData, out testData);
      Console.WriteLine("Done\n");

      Console.WriteLine("Training data:");
      DataUtils.ShowMatrix(trainData, 4, 2, true);
      Console.WriteLine("Test data:");
      DataUtils.ShowMatrix(testData, 4, 2, true);

      int maxEpochs = 1000;
      double learnRate = 0.05;
      double momentum = 0.01;

      // -------------------------------------------------------------
      // EXPERIMENT 1: Baseline (Mean Squared Error, No Weight Decay)
      // -------------------------------------------------------------
      Console.WriteLine("\n=======================================================");
      Console.WriteLine(" EXPERIMENT 1: Baseline (Mean Squared Error, No Decay)");
      Console.WriteLine("=======================================================");
      Console.WriteLine("Creating a " + numInput + "-" + numHidden +
        "-" + numOutput + " neural network");
      NeuralNetwork nn1 = new NeuralNetwork(numInput, numHidden, numOutput);

      Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
      Console.WriteLine("Setting learnRate = " + learnRate.ToString("F2"));
      Console.WriteLine("Setting momentum  = " + momentum.ToString("F2"));
      Console.WriteLine("Setting errorType = Mean Squared Error");
      Console.WriteLine("Setting decayRate = 0.00 (No Weight Decay)");

      Console.WriteLine("\nStarting training");
      double[] weights1 = nn1.Train(trainData, maxEpochs, learnRate, momentum,
        ErrorType.MeanSquared, 0.0);
      Console.WriteLine("Done");
      Console.WriteLine("\nFinal neural network model weights and biases:\n");
      DataUtils.ShowVector(weights1, 2, 10, true);

      double trainErr1 = nn1.MeanSquaredError(trainData);
      double trainAcc1 = nn1.Accuracy(trainData);
      double testAcc1 = nn1.Accuracy(testData);
      double weightNorm1 = nn1.WeightNorm();
      Console.WriteLine("\nFinal MSE on training data      = " + trainErr1.ToString("F4"));
      Console.WriteLine("Final weight norm (L2 norm)     = " + weightNorm1.ToString("F4"));
      Console.WriteLine("Final accuracy on training data = " + trainAcc1.ToString("F4"));
      Console.WriteLine("Final accuracy on test data     = " + testAcc1.ToString("F4"));

      // -------------------------------------------------------------
      // EXPERIMENT 2: Cross-Entropy Error (No Weight Decay)
      // -------------------------------------------------------------
      Console.WriteLine("\n=======================================================");
      Console.WriteLine(" EXPERIMENT 2: Cross-Entropy Error (No Weight Decay)");
      Console.WriteLine("=======================================================");
      Console.WriteLine("Creating a " + numInput + "-" + numHidden +
        "-" + numOutput + " neural network");
      NeuralNetwork nn2 = new NeuralNetwork(numInput, numHidden, numOutput);

      Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
      Console.WriteLine("Setting learnRate = " + learnRate.ToString("F2"));
      Console.WriteLine("Setting momentum  = " + momentum.ToString("F2"));
      Console.WriteLine("Setting errorType = Cross-Entropy Error");
      Console.WriteLine("Setting decayRate = 0.00 (No Weight Decay)");

      Console.WriteLine("\nStarting training");
      double[] weights2 = nn2.Train(trainData, maxEpochs, learnRate, momentum,
        ErrorType.CrossEntropy, 0.0);
      Console.WriteLine("Done");
      Console.WriteLine("\nFinal neural network model weights and biases:\n");
      DataUtils.ShowVector(weights2, 2, 10, true);

      double trainErr2 = nn2.CrossEntropyError(trainData);
      double trainAcc2 = nn2.Accuracy(trainData);
      double testAcc2 = nn2.Accuracy(testData);
      double weightNorm2 = nn2.WeightNorm();
      Console.WriteLine("\nFinal Cross-Entropy on train    = " + trainErr2.ToString("F4"));
      Console.WriteLine("Final weight norm (L2 norm)     = " + weightNorm2.ToString("F4"));
      Console.WriteLine("Final accuracy on training data = " + trainAcc2.ToString("F4"));
      Console.WriteLine("Final accuracy on test data     = " + testAcc2.ToString("F4"));

      // -------------------------------------------------------------
      // EXPERIMENT 3: Cross-Entropy Error WITH Weight Decay
      // -------------------------------------------------------------
      double decayRate = 0.001;
      Console.WriteLine("\n=======================================================");
      Console.WriteLine(" EXPERIMENT 3: Cross-Entropy WITH Weight Decay (Regularization)");
      Console.WriteLine("=======================================================");
      Console.WriteLine("Creating a " + numInput + "-" + numHidden +
        "-" + numOutput + " neural network");
      NeuralNetwork nn3 = new NeuralNetwork(numInput, numHidden, numOutput);

      Console.WriteLine("\nSetting maxEpochs = " + maxEpochs);
      Console.WriteLine("Setting learnRate = " + learnRate.ToString("F2"));
      Console.WriteLine("Setting momentum  = " + momentum.ToString("F2"));
      Console.WriteLine("Setting errorType = Cross-Entropy Error");
      Console.WriteLine("Setting decayRate = " + decayRate.ToString("F4") + " (L2 Regularization)");

      Console.WriteLine("\nStarting training");
      double[] weights3 = nn3.Train(trainData, maxEpochs, learnRate, momentum,
        ErrorType.CrossEntropy, decayRate);
      Console.WriteLine("Done");
      Console.WriteLine("\nFinal neural network model weights and biases:\n");
      DataUtils.ShowVector(weights3, 2, 10, true);

      double trainErr3 = nn3.CrossEntropyError(trainData);
      double trainAcc3 = nn3.Accuracy(trainData);
      double testAcc3 = nn3.Accuracy(testData);
      double weightNorm3 = nn3.WeightNorm();
      Console.WriteLine("\nFinal Cross-Entropy on train    = " + trainErr3.ToString("F4"));
      Console.WriteLine("Final weight norm (L2 norm)     = " + weightNorm3.ToString("F4"));
      Console.WriteLine("Final accuracy on training data = " + trainAcc3.ToString("F4"));
      Console.WriteLine("Final accuracy on test data     = " + testAcc3.ToString("F4"));

      // -------------------------------------------------------------
      // COMPARISON SUMMARY TABLE
      // -------------------------------------------------------------
      Console.WriteLine("\n==========================================================================");
      Console.WriteLine(" COMPARISON OF BACK-PROPAGATION CONFIGURATIONS");
      Console.WriteLine("==========================================================================");
      Console.WriteLine("{0,-32} {1,-12} {2,-14} {3,-12} {4,-12}",
        "Configuration", "Train Loss", "Weight Norm", "Train Acc", "Test Acc");
      Console.WriteLine(new string('-', 84));
      Console.WriteLine("{0,-32} {1,-12:F4} {2,-14:F4} {3,-12:P2} {4,-12:P2}",
        "1. Baseline (MSE, No Decay)", trainErr1, weightNorm1, trainAcc1, testAcc1);
      Console.WriteLine("{0,-32} {1,-12:F4} {2,-14:F4} {3,-12:P2} {4,-12:P2}",
        "2. Cross-Entropy (No Decay)", trainErr2, weightNorm2, trainAcc2, testAcc2);
      Console.WriteLine("{0,-32} {1,-12:F4} {2,-14:F4} {3,-12:P2} {4,-12:P2}",
        "3. Cross-Entropy + Weight Decay", trainErr3, weightNorm3, trainAcc3, testAcc3);
      Console.WriteLine(new string('-', 84));
      Console.WriteLine("\nKey Observations:");
      Console.WriteLine("  1. Cross-Entropy avoids the softmax derivative saturation term (1-y)*y,");
      Console.WriteLine("     yielding faster learning signals for classification.");
      Console.WriteLine("  2. Weight Decay penalizes large weights, significantly reducing the");
      Console.WriteLine("     L2 weight norm (" + weightNorm2.ToString("F2") + " -> " + weightNorm3.ToString("F2") + "), helping prevent over-fitting.");
      Console.WriteLine("==========================================================================\n");

      Console.WriteLine("End back-propagation demo\n");
      Console.ReadLine();
    }
  }
}
