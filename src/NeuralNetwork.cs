using System;

namespace CodingBackProp
{
  /// <summary>
  /// 3-layer fully connected neural network trained with back-propagation.
  /// Supports MSE and Cross-Entropy loss, with optional L2 weight decay.
  /// </summary>
  public class NeuralNetwork
  {
    private readonly int numInput;
    private readonly int numHidden;
    private readonly int numOutput;

    private readonly double[] inputs;
    private readonly double[][] ihWeights; // input → hidden
    private readonly double[] hBiases;
    private readonly double[] hOutputs;

    private readonly double[][] hoWeights; // hidden → output
    private readonly double[] oBiases;
    private readonly double[] outputs;

    private readonly Random rnd;

    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      inputs = new double[numInput];

      ihWeights = MakeMatrix(numInput, numHidden, 0.0);
      hBiases = new double[numHidden];
      hOutputs = new double[numHidden];

      hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
      oBiases = new double[numOutput];
      outputs = new double[numOutput];

      rnd = new Random(0);
      InitializeWeights();
    }

    private static double[][] MakeMatrix(int rows, int cols, double v)
    {
      var result = new double[rows][];
      for (int r = 0; r < result.Length; ++r)
        result[r] = new double[cols];
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = v;
      return result;
    }

    private void InitializeWeights()
    {
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      var initialWeights = new double[numWeights];
      for (int i = 0; i < initialWeights.Length; ++i)
        initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
      SetWeights(initialWeights);
    }

    public void SetWeights(double[] weights)
    {
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
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
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      var result = new double[numWeights];
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
      var hSums = new double[numHidden];
      var oSums = new double[numOutput];

      for (int i = 0; i < xValues.Length; ++i)
        inputs[i] = xValues[i];

      // accumulate weighted inputs into each hidden node
      for (int j = 0; j < numHidden; ++j)
        for (int i = 0; i < numInput; ++i)
          hSums[j] += inputs[i] * ihWeights[i][j];

      for (int i = 0; i < numHidden; ++i)
        hSums[i] += hBiases[i];

      for (int i = 0; i < numHidden; ++i)
        hOutputs[i] = HyperTan(hSums[i]);

      // accumulate hidden outputs into each output node
      for (int j = 0; j < numOutput; ++j)
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += hOutputs[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)
        oSums[i] += oBiases[i];

      double[] softOut = Softmax(oSums);
      Array.Copy(softOut, outputs, softOut.Length);

      var retResult = new double[numOutput];
      Array.Copy(outputs, retResult, retResult.Length);
      return retResult;
    }

    // clamps extreme values to avoid overflow in tanh
    private static double HyperTan(double x)
    {
      if (x < -20.0) return -1.0;
      if (x > 20.0) return 1.0;
      return Math.Tanh(x);
    }

    private static double[] Softmax(double[] oSums)
    {
      double sum = 0.0;
      for (int i = 0; i < oSums.Length; ++i)
        sum += Math.Exp(oSums[i]);

      var result = new double[oSums.Length];
      for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i]) / sum;

      return result;
    }

    /// <summary>Backwards-compatible overload — defaults to MSE with no weight decay.</summary>
    public double[] Train(double[][] trainData, int maxEpochs, double learnRate, double momentum) =>
      Train(trainData, maxEpochs, learnRate, momentum, ErrorType.MeanSquared, 0.0);

    /// <summary>Trains the network using the specified loss function and optional L2 weight decay.</summary>
    public double[] Train(double[][] trainData, int maxEpochs, double learnRate, double momentum,
      ErrorType errorType, double decayRate)
    {
      var hoGrads = MakeMatrix(numHidden, numOutput, 0.0);
      var obGrads = new double[numOutput];

      var ihGrads = MakeMatrix(numInput, numHidden, 0.0);
      var hbGrads = new double[numHidden];

      var oSignals = new double[numOutput];
      var hSignals = new double[numHidden];

      // previous deltas for momentum
      var ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
      var hPrevBiasesDelta = new double[numHidden];
      var hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
      var oPrevBiasesDelta = new double[numOutput];

      int epoch = 0;
      var xValues = new double[numInput];
      var tValues = new double[numOutput];
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
          Console.WriteLine($"epoch = {epoch}  error = {trainErr:F4}");
        }

        Shuffle(sequence);
        for (int ii = 0; ii < trainData.Length; ++ii)
        {
          int idx = sequence[ii];
          Array.Copy(trainData[idx], xValues, numInput);
          Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
          ComputeOutputs(xValues);

          // step 1 — output node error signals
          for (int k = 0; k < numOutput; ++k)
          {
            errorSignal = tValues[k] - outputs[k];
            if (errorType == ErrorType.CrossEntropy)
            {
              // with softmax + cross-entropy the derivative simplifies to (t - y),
              // so we skip the extra softmax factor used in MSE
              oSignals[k] = errorSignal;
            }
            else
            {
              derivative = (1 - outputs[k]) * outputs[k]; // softmax derivative
              oSignals[k] = errorSignal * derivative;
            }
          }

          // step 2 — hidden-to-output weight gradients
          for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
              hoGrads[j][k] = oSignals[k] * hOutputs[j];

          // step 2b — output bias gradients
          for (int k = 0; k < numOutput; ++k)
            obGrads[k] = oSignals[k] * 1.0;

          // step 3 — hidden node error signals (back-propagate through tanh)
          for (int j = 0; j < numHidden; ++j)
          {
            derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // tanh derivative
            double sum = 0.0;
            for (int k = 0; k < numOutput; ++k)
              sum += oSignals[k] * hoWeights[j][k];
            hSignals[j] = derivative * sum;
          }

          // step 4 — input-to-hidden weight gradients
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihGrads[i][j] = hSignals[j] * inputs[i];

          // step 4b — hidden bias gradients
          for (int j = 0; j < numHidden; ++j)
            hbGrads[j] = hSignals[j] * 1.0;

          // --- weight and bias updates ---

          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              double delta = ihGrads[i][j] * learnRate;
              ihWeights[i][j] += delta;
              if (decayRate > 0.0)
                ihWeights[i][j] -= decayRate * ihWeights[i][j]; // L2 shrinkage
              ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
              ihPrevWeightsDelta[i][j] = delta;
            }
          }

          for (int j = 0; j < numHidden; ++j)
          {
            double delta = hbGrads[j] * learnRate;
            hBiases[j] += delta;
            hBiases[j] += hPrevBiasesDelta[j] * momentum;
            hPrevBiasesDelta[j] = delta;
          }

          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              double delta = hoGrads[j][k] * learnRate;
              hoWeights[j][k] += delta;
              if (decayRate > 0.0)
                hoWeights[j][k] -= decayRate * hoWeights[j][k]; // L2 shrinkage
              hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
              hoPrevWeightsDelta[j][k] = delta;
            }
          }

          for (int k = 0; k < numOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
            oBiases[k] += oPrevBiasesDelta[k] * momentum;
            oPrevBiasesDelta[k] = delta;
          }
        }
      }

      return GetWeights();
    }

    private void Shuffle(int[] sequence)
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = rnd.Next(i, sequence.Length);
        (sequence[i], sequence[r]) = (sequence[r], sequence[i]);
      }
    }

    public double Error(double[][] trainData, ErrorType errorType = ErrorType.MeanSquared) =>
      errorType == ErrorType.CrossEntropy
        ? CrossEntropyError(trainData)
        : MeanSquaredError(trainData);

    public double MeanSquaredError(double[][] trainData)
    {
      double sumSquaredError = 0.0;
      var xValues = new double[numInput];
      var tValues = new double[numOutput];

      for (int i = 0; i < trainData.Length; ++i)
      {
        Array.Copy(trainData[i], xValues, numInput);
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput);
        double[] yValues = ComputeOutputs(xValues);
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
      var xValues = new double[numInput];
      var tValues = new double[numOutput];

      for (int i = 0; i < trainData.Length; ++i)
      {
        Array.Copy(trainData[i], xValues, numInput);
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput);
        double[] yValues = ComputeOutputs(xValues);
        for (int j = 0; j < numOutput; ++j)
        {
          if (tValues[j] > 0.0)
          {
            // clamp to avoid log(0)
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
      var xValues = new double[numInput];
      var tValues = new double[numOutput];

      for (int i = 0; i < testData.Length; ++i)
      {
        Array.Copy(testData[i], xValues, numInput);
        Array.Copy(testData[i], numInput, tValues, 0, numOutput);
        double[] yValues = ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues);
        int tMaxIndex = MaxIndex(tValues);

        if (maxIndex == tMaxIndex)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (double)numCorrect / (numCorrect + numWrong);
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
