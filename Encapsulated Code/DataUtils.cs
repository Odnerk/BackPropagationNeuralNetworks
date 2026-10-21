using System;

namespace CodingBackProp
{
  /// <summary>
  /// Utility methods for matrix/vector display, synthetic data generation, and dataset splitting.
  /// </summary>
  public static class DataUtils
  {
    public static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool indices)
    {
      int len = matrix.Length.ToString().Length;
      for (int i = 0; i < numRows; ++i)
      {
        if (indices == true)
          Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < matrix[i].Length; ++j)
        {
          double v = matrix[i][j];
          if (v >= 0.0)
            Console.Write(" ");
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
        Console.WriteLine("");
      }

      if (numRows < matrix.Length)
      {
        Console.WriteLine(". . .");
        int lastRow = matrix.Length - 1;
        if (indices == true)
          Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < matrix[lastRow].Length; ++j)
        {
          double v = matrix[lastRow][j];
          if (v >= 0.0)
            Console.Write(" ");
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
      }
      Console.WriteLine("\n");
    }

    public static void ShowVector(double[] vector, int decimals, int lineLen, bool newLine)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i > 0 && i % lineLen == 0) Console.WriteLine("");
        if (vector[i] >= 0) Console.Write(" ");
        Console.Write(vector[i].ToString("F" + decimals) + " ");
      }
      if (newLine == true)
        Console.WriteLine("");
    }

    public static double[][] MakeAllData(int numInput, int numHidden, int numOutput, int numRows, int seed)
    {
      Random rnd = new Random(seed);
      int numWeights = (numInput * numHidden) + numHidden +
        (numHidden * numOutput) + numOutput;
      double[] weights = new double[numWeights]; // actually weights & biases
      for (int i = 0; i < numWeights; ++i)
        weights[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to 10.0]

      Console.WriteLine("Generating weights and biases:");
      ShowVector(weights, 2, 10, true);

      double[][] result = new double[numRows][]; // allocate return-result
      for (int i = 0; i < numRows; ++i)
        result[i] = new double[numInput + numOutput]; // 1-of-N in last column

      NeuralNetwork gnn = new NeuralNetwork(numInput, numHidden, numOutput); // generating NN
      gnn.SetWeights(weights);

      for (int r = 0; r < numRows; ++r) // for each row
      {
        // generate random inputs
        double[] inputs = new double[numInput];
        for (int i = 0; i < numInput; ++i)
          inputs[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to 10.0]

        // compute outputs
        double[] outputs = gnn.ComputeOutputs(inputs);

        // translate outputs to 1-of-N
        double[] oneOfN = new double[numOutput]; // all 0.0

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

        // place inputs and 1-of-N output values into curr row
        int c = 0; // column into result[][]
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
      Random rnd = new Random(seed);
      int totRows = allData.Length;
      int numTrainRows = (int)(totRows * trainPct); // usually 0.80
      int numTestRows = totRows - numTrainRows;
      trainData = new double[numTrainRows][];
      testData = new double[numTestRows][];

      double[][] copy = new double[allData.Length][]; // ref copy of data
      for (int i = 0; i < copy.Length; ++i)
        copy[i] = allData[i];

      for (int i = 0; i < copy.Length; ++i) // scramble order
      {
        int r = rnd.Next(i, copy.Length); // use Fisher-Yates
        double[] tmp = copy[r];
        copy[r] = copy[i];
        copy[i] = tmp;
      }
      for (int i = 0; i < numTrainRows; ++i)
        trainData[i] = copy[i];

      for (int i = 0; i < numTestRows; ++i)
        testData[i] = copy[i + numTrainRows];
    }
  }
}
