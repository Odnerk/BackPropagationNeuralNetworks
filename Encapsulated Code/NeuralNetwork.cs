using System;

namespace CodingBackProp
{
  /// <summary>
  /// A 3-layer fully connected neural network trained with back-propagation.
  /// Supports Mean Squared Error and Cross-Entropy loss functions, with optional L2 Weight Decay (regularization).
  /// </summary>
  public class NeuralNetwork
  {
    private int numInput; // number input nodes
    private int numHidden;
    private int numOutput;

    private double[] inputs;
    private double[][] ihWeights; // input-hidden
    private double[] hBiases;
    private double[] hOutputs;

    private double[][] hoWeights; // hidden-output
    private double[] oBiases;
    private double[] outputs;

    private Random rnd;

    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      this.inputs = new double[numInput];

      this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
      this.hBiases = new double[numHidden];
      this.hOutputs = new double[numHidden];

      this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
      this.oBiases = new double[numOutput];
      this.outputs = new double[numOutput];

      this.rnd = new Random(0);
      this.InitializeWeights();
    }

    private static double[][] MakeMatrix(int rows, int cols, double v)
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < result.Length; ++r)
        result[r] = new double[cols];
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = v;
      return result;
    }

    private void InitializeWeights()
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] initialWeights = new double[numWeights];
      for (int i = 0; i < initialWeights.Length; ++i)
        initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
      this.SetWeights(initialWeights);
    }

    public void SetWeights(double[] weights)
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      if (weights.Length != numWeights)
        throw new Exception("Bad weights array in SetWeights");

      int k = 0;

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];
      for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
    }

    public double[] GetWeights()
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] result = new double[numWeights];
      int k = 0;
      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < hBiases.Length; ++i)
        result[k++] = hBiases[i];
      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < oBiases.Length; ++i)
        result[k++] = oBiases[i];
      return result;
    }

    public double[] ComputeOutputs(double[] xValues)
    {
      double[] hSums = new double[numHidden];
      double[] oSums = new double[numOutput];

      for (int i = 0; i < xValues.Length; ++i)
        this.inputs[i] = xValues[i];

      for (int j = 0; j < numHidden; ++j)
        for (int i = 0; i < numInput; ++i)
          hSums[j] += this.inputs[i] * this.ihWeights[i][j];

      for (int i = 0; i < numHidden; ++i)
        hSums[i] += this.hBiases[i];

      for (int i = 0; i < numHidden; ++i)
        this.hOutputs[i] = HyperTan(hSums[i]);

      for (int j = 0; j < numOutput; ++j)
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += hOutputs[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)
        oSums[i] += oBiases[i];

      double[] softOut = Softmax(oSums);
      Array.Copy(softOut, outputs, softOut.Length);

      double[] retResult = new double[numOutput];
      Array.Copy(this.outputs, retResult, retResult.Length);
      return retResult;
    }

    private static double HyperTan(double x)
    {
      if (x < -20.0) return -1.0;
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
    }

    private static double[] Softmax(double[] oSums)
    {
      double sum = 0.0;
      for (int i = 0; i < oSums.Length; ++i)
        sum += Math.Exp(oSums[i]);

      double[] result = new double[oSums.Length];
      for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i]) / sum;

      return result;
    }

    // Overload for backwards compatibility (Baseline MSE, no weight decay)
    public double[] Train(double[][] trainData, int maxEpochs, double learnRate, double momentum)
    {
      return Train(trainData, maxEpochs, learnRate, momentum, ErrorType.MeanSquared, 0.0);
    }

    // Full Train method supporting ErrorType and L2 Regularization (Weight Decay)
    public double[] Train(double[][] trainData, int maxEpochs, double learnRate, double momentum,
      ErrorType errorType, double decayRate)
    {
      double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0);
      double[] obGrads = new double[numOutput];

      double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);
      double[] hbGrads = new double[numHidden];

      double[] oSignals = new double[numOutput];
      double[] hSignals = new double[numHidden];

      double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
      double[] hPrevBiasesDelta = new double[numHidden];
      double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
      double[] oPrevBiasesDelta = new double[numOutput];

      int epoch = 0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];
      double derivative = 0.0;
      double errorSignal = 0.0;

      int[] sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      int errInterval = maxEpochs / 10;
      while (epoch < maxEpochs)
      {
        ++epoch;

        if (epoch % errInterval == 0 && epoch < maxEpochs)
        {
          double trainErr = Error(trainData, errorType);
          Console.WriteLine("epoch = " + epoch + "  error = " + trainErr.ToString("F4"));
        }

        Shuffle(sequence);
        for (int ii = 0; ii < trainData.Length; ++ii)
        {
          int idx = sequence[ii];
          Array.Copy(trainData[idx], xValues, numInput);
          Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
          ComputeOutputs(xValues);

          // 1. Compute output node signals
          for (int k = 0; k < numOutput; ++k)
          {
            errorSignal = tValues[k] - outputs[k];
            if (errorType == ErrorType.CrossEntropy)
            {
              // For Cross-Entropy loss paired with Softmax activation,
              // dE/dz_k = outputs[k] - tValues[k] (or t - y with our update sign convention)
              oSignals[k] = errorSignal;
            }
            else // MeanSquared
            {
              derivative = (1 - outputs[k]) * outputs[k]; // for softmax with MSE
              oSignals[k] = errorSignal * derivative;
            }
          }

          // 2. Compute hidden-to-output weight gradients
          for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
              hoGrads[j][k] = oSignals[k] * hOutputs[j];

          // 2b. Compute output bias gradients
          for (int k = 0; k < numOutput; ++k)
            obGrads[k] = oSignals[k] * 1.0;

          // 3. Compute hidden node signals
          for (int j = 0; j < numHidden; ++j)
          {
            derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // tanh derivative
            double sum = 0.0;
            for (int k = 0; k < numOutput; ++k)
              sum += oSignals[k] * hoWeights[j][k];
            hSignals[j] = derivative * sum;
          }

          // 4. Compute input-hidden weight gradients
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihGrads[i][j] = hSignals[j] * inputs[i];

          // 4b. Compute hidden node bias gradients
          for (int j = 0; j < numHidden; ++j)
            hbGrads[j] = hSignals[j] * 1.0;

          // == Update weights and biases ==

          // Update input-to-hidden weights
          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              double delta = ihGrads[i][j] * learnRate;
              ihWeights[i][j] += delta;
              if (decayRate > 0.0)
                ihWeights[i][j] -= (decayRate * ihWeights[i][j]); // L2 weight decay
              ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
              ihPrevWeightsDelta[i][j] = delta;
            }
          }

          // Update hidden biases
          for (int j = 0; j < numHidden; ++j)
          {
            double delta = hbGrads[j] * learnRate;
            hBiases[j] += delta;
            hBiases[j] += hPrevBiasesDelta[j] * momentum;
            hPrevBiasesDelta[j] = delta;
          }

          // Update hidden-to-output weights
          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              double delta = hoGrads[j][k] * learnRate;
              hoWeights[j][k] += delta;
              if (decayRate > 0.0)
                hoWeights[j][k] -= (decayRate * hoWeights[j][k]); // L2 weight decay
              hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
              hoPrevWeightsDelta[j][k] = delta;
            }
          }

          // Update output node biases
          for (int k = 0; k < numOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
            oBiases[k] += oPrevBiasesDelta[k] * momentum;
            oPrevBiasesDelta[k] = delta;
          }
        }
      }

      double[] bestWts = GetWeights();
      return bestWts;
    }

    private void Shuffle(int[] sequence)
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = this.rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    }

    public double Error(double[][] trainData, ErrorType errorType = ErrorType.MeanSquared)
    {
      if (errorType == ErrorType.CrossEntropy)
        return CrossEntropyError(trainData);
      else
        return MeanSquaredError(trainData);
    }

    public double MeanSquaredError(double[][] trainData)
    {
      double sumSquaredError = 0.0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];

      for (int i = 0; i < trainData.Length; ++i)
      {
        Array.Copy(trainData[i], xValues, numInput);
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput);
        double[] yValues = this.ComputeOutputs(xValues);
        for (int j = 0; j < numOutput; ++j)
        {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }
      return sumSquaredError / trainData.Length;
    }

    public double CrossEntropyError(double[][] trainData)
    {
      double sumCrossError = 0.0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];

      for (int i = 0; i < trainData.Length; ++i)
      {
        Array.Copy(trainData[i], xValues, numInput);
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput);
        double[] yValues = this.ComputeOutputs(xValues);
        for (int j = 0; j < numOutput; ++j)
        {
          if (tValues[j] > 0.0)
          {
            double prob = Math.Max(yValues[j], 1.0e-15);
            sumCrossError += Math.Log(prob);
          }
        }
      }
      return -1.0 * sumCrossError / trainData.Length;
    }

    public double WeightNorm()
    {
      double sumSq = 0.0;
      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          sumSq += ihWeights[i][j] * ihWeights[i][j];

      for (int j = 0; j < numHidden; ++j)
        for (int k = 0; k < numOutput; ++k)
          sumSq += hoWeights[j][k] * hoWeights[j][k];

      return Math.Sqrt(sumSq);
    }

    public double Accuracy(double[][] testData)
    {
      int numCorrect = 0;
      int numWrong = 0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];
      double[] yValues;

      for (int i = 0; i < testData.Length; ++i)
      {
        Array.Copy(testData[i], xValues, numInput);
        Array.Copy(testData[i], numInput, tValues, 0, numOutput);
        yValues = this.ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues);
        int tMaxIndex = MaxIndex(tValues);

        if (maxIndex == tMaxIndex)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

    private static int MaxIndex(double[] vector)
    {
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 0; i < vector.Length; ++i)
      {
        if (vector[i] > biggestVal)
        {
          biggestVal = vector[i];
          bigIndex = i;
        }
      }
      return bigIndex;
    }
  }
}
