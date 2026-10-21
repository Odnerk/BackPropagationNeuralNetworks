using System;

namespace CodingBackProp
{
  /// <summary>
  /// Helper methods for printing matrices and vectors, generating synthetic data, and splitting datasets.
  /// </summary>
  public static class DataUtils
  {
    public static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool indices)
    {
      int len = matrix.Length.ToString().Length;
      for (int i = 0; i < numRows; ++i)
      {
        if (indices)
          Console.Write($"[{i.ToString().PadLeft(len)}]  ");
        for (int j = 0; j < matrix[i].Length; ++j)
        {
          double v = matrix[i][j];
          if (v >= 0.0)
            Console.Write(" ");
          Console.Write($"{v.ToString("F" + decimals)}  ");
        }
        Console.WriteLine();
      }

      if (numRows < matrix.Length)
      {
        Console.WriteLine(". . .");
        int lastRow = matrix.Length - 1;
        if (indices)
          Console.Write($"[{lastRow.ToString().PadLeft(len)}]  ");
        for (int j = 0; j < matrix[lastRow].Length; ++j)
        {
          double v = matrix[lastRow][j];
          if (v >= 0.0)
            Console.Write(" ");
          Console.Write($"{v.ToString("F" + decimals)}  ");
        }
      }
      Console.WriteLine("\n");
    }

    public static void ShowVector(double[] vector, int decimals, int lineLen, bool newLine)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i > 0 && i % lineLen == 0) Console.WriteLine();
        if (vector[i] >= 0) Console.Write(" ");
        Console.Write($"{vector[i].ToString("F" + decimals)} ");
      }
      if (newLine)
        Console.WriteLine();
    }

    public static double[][] MakeAllData(int numInput, int numHidden, int numOutput, int numRows, int seed)
    {
      var rnd = new Random(seed);
      int numWeights = (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;
      var weights = new double[numWeights];
      for (int i = 0; i < numWeights; ++i)
        weights[i] = 20.0 * rnd.NextDouble() - 10.0;

      Console.WriteLine("Generating weights and biases:");
      ShowVector(weights, 2, 10, true);

      var result = new double[numRows][];
      for (int i = 0; i < numRows; ++i)
        result[i] = new double[numInput + numOutput];

      var gnn = new NeuralNetwork(numInput, numHidden, numOutput);
      gnn.SetWeights(weights);

      for (int r = 0; r < numRows; ++r)
      {
        var inputs = new double[numInput];
        for (int i = 0; i < numInput; ++i)
          inputs[i] = 20.0 * rnd.NextDouble() - 10.0;

        double[] outputs = gnn.ComputeOutputs(inputs);

        // find the winning class and encode as one-hot
        var oneOfN = new double[numOutput];
        int maxIndex = 0;
        double maxValue = outputs[0];
        for (int i = 0; i < numOutput; ++i)
        {
          if (outputs[i] > maxValue)
          {
            maxIndex = i;
            maxValue = outputs[i];
          }
        }
        oneOfN[maxIndex] = 1.0;

        int c = 0;
        for (int i = 0; i < numInput; ++i)
          result[r][c++] = inputs[i];
        for (int i = 0; i < numOutput; ++i)
          result[r][c++] = oneOfN[i];
      }
      return result;
    }

    public static void SplitTrainTest(double[][] allData, double trainPct, int seed,
      out double[][] trainData, out double[][] testData)
    {
      var rnd = new Random(seed);
      int totRows = allData.Length;
      int numTrainRows = (int)(totRows * trainPct);
      int numTestRows = totRows - numTrainRows;
      trainData = new double[numTrainRows][];
      testData = new double[numTestRows][];

      // shallow-copy so we can shuffle without touching the original
      var copy = new double[allData.Length][];
      for (int i = 0; i < copy.Length; ++i)
        copy[i] = allData[i];

      // Fisher-Yates shuffle
      for (int i = 0; i < copy.Length; ++i)
      {
        int r = rnd.Next(i, copy.Length);
        (copy[i], copy[r]) = (copy[r], copy[i]);
      }
      for (int i = 0; i < numTrainRows; ++i)
        trainData[i] = copy[i];

      for (int i = 0; i < numTestRows; ++i)
        testData[i] = copy[i + numTrainRows];
    }
  }
}
