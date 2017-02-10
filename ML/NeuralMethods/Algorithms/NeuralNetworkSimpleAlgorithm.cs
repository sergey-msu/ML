using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core.Mathematics;
using ML.Core;

namespace ML.NeuralMethods.Algorithms
{
  public class NeuralNetworkSimpleAlgorithm : NeuralNetworkAlgorithmBase
  {
    public NeuralNetworkSimpleAlgorithm(ClassifiedSample classifiedSample)
      : base(classifiedSample)
    {
    }

    public override string ID   { get { return "NNET-SIMP"; } }
    public override string Name { get { return "Simple Neural Network"; } }

    public int    EpochCount { get; set; }
    public double Margin     { get; set; }
    public double Step       { get; set; }


    public void Train()
    {
      Network.Epoch = 0;

      var weightCount = Network.Layers.Sum(l => l.Neurons.Sum(n => n.WeightCount));
      var weights = new double[weightCount];

      for (int i=0; i<EpochCount; i++) // epoch
      {
        Network.Epoch++;

        foreach (var pdata in TrainingSample) // iteration
        {
          var point = pdata.Key;
          var cls = pdata.Value;

          for (int k=0; k<weightCount; k++)
            minimizeError(point, cls, weightCount, k);
        }
      }
    }

    #region .pvt

    private void minimizeError(Point data, Class cls, int count, int idx)
    {
      var deltas = new double[count];
      deltas[idx] = -Margin;
      Network.UpdateWeights(deltas, true);
      var minError = calculateError(data, cls);
      var minDx = -Margin;

      deltas[idx] = Step;
      var dx = -Margin;
      while (dx < Margin)
      {
        Network.UpdateWeights(deltas, true);
        var error = calculateError(data, cls);
        if (error < minError)
        {
          minError = error;
          minDx = dx;
        }
        dx += Step;
      }

      deltas[idx] = minDx-dx;
      Network.UpdateWeights(deltas, true);
    }

    private double calculateError(Point data, Class cls)
    {
      var result = Network.Calculate(data);

      int idx;
      double max;
      MathUtils.CalcMax(result, out idx, out max);
      var calcClsVal = result[idx];
      var expClsVal = result[cls.Order];

      var error = Math.Abs(calcClsVal - expClsVal);

      return error;
    }

    #endregion
  }
}
