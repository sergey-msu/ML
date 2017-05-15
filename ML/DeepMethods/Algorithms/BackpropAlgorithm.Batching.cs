using System;
using ML.Core;

namespace ML.DeepMethods.Algorithms
{
  /// <summary>
  /// Batch CPU parallelization
  /// </summary>
  public partial class BackpropAlgorithm : ConvNetAlgorithmBase
  {
    #region Inner

    private class BatchContext
    {
      #region Inner

        private class BatchItem
        {
          private BackpropAlgorithm m_Algorithm;
          private bool m_IsBusy;
          private double[][][,] m_Values;
          private double[][][,] m_Errors;
          private double[][]    m_GradientPortion;

          private object m_Sync = new object();

          public BatchItem(BackpropAlgorithm alg)
          {
            m_Algorithm = alg;
            var net = alg.Net;

            m_Values = new double[net.LayerCount][][,];
            m_Errors = new double[net.LayerCount][][,];
            m_GradientPortion = new double[net.LayerCount][];

            for (int l=0; l<net.LayerCount; l++)
            {
              var layer = net[l];

              m_GradientPortion[l] = new double[layer.ParamCount];
              m_Values[l] = new double[layer.OutputDepth][,];
              m_Errors[l] = new double[layer.OutputDepth][,];
              for (int p=0; p<layer.OutputDepth; p++)
              {
                m_Values[l][p] = new double[layer.OutputHeight, layer.OutputWidth];
                m_Errors[l][p] = new double[layer.OutputHeight, layer.OutputWidth];
              }
            }
          }

          public bool IsBusy { get { return m_IsBusy; } }

          public bool DoIter(double[][,] input, Class cls)
          {
            if (m_IsBusy) return false;

            lock (m_Sync)
            {
              if (m_IsBusy) return false;
              m_IsBusy = true;
            }

            try
            {
              //lock (m_Algorithm) { System.Threading.Thread.Sleep(2000); }
              m_Algorithm.runIteration(m_Values, m_Errors, m_GradientPortion, input, cls);
            }
            finally
            {
              m_IsBusy = false;
            }
            return true;
          }
      }

      #endregion

      private BackpropAlgorithm m_Algorithm;
      private BatchItem[] m_Items;
      private int         m_Threads;
      private object      m_BatchItemSync = new object();

      public BatchContext(BackpropAlgorithm alg, int maxDegreesOfParallelism)
      {
        m_Algorithm = alg;
        m_Threads = maxDegreesOfParallelism;
        m_Items = new BatchItem[maxDegreesOfParallelism];
      }

      public void Push(double[][,] data, Class cls)
      {
        int i = -1;
        while (true)
        {
          i++;
          var j = i % m_Threads;
          var item = m_Items[j];
          if (item == null)
          {
            lock (m_BatchItemSync)
            {
              item = m_Items[j];
              if (item == null)
              {
                item = new BatchItem(m_Algorithm);
                m_Items[j] = item;
              }
            }
          }

          if (!item.DoIter(data, cls)) continue;
          return;
        }
      }
    }

    #endregion

    #region .pvt

    private void runIteration(double[][][,] values, double[][][,] errors, double[][] gradientPortions,
                              double[][,] input, Class cls)
    {
      // feed forward
      var iterLoss = feedForward(values, errors, input, cls);

      // feed backward
      var lcount = Net.LayerCount;
      for (int i=lcount-1; i>=0; i--)
      {
        var layer  = Net[i];
        var error  = errors[i];
        var player = Net[i-1];
        var pvalue = (i>0) ? values[i-1] : input;
        var perror = (i>0) ? errors[i-1] : null;
        var gradientPortion = gradientPortions[i];

        // error backpropagation
        if (i>0)
          layer.Backprop(player, pvalue, perror, error);

        // prepare gradient updates
        layer.SetLayerGradient(pvalue, error, gradientPortion, false);
      }

      // update gradient with iteration portion
      lock (m_Gradient)
      {
        for (int i=lcount-1; i>=0; i--)
        {
          var gradient = m_Gradient[i];
          var gradientPortion = gradientPortions[i];
          var glen = gradient.Length;
          for (int j=0; j<glen; j++)
            gradient[j] += gradientPortion[j];
        }

        m_IterLossValue += iterLoss;
      }
    }

    private double feedForward(double[][][,] values, double[][][,] errors, double[][,] input, Class cls)
    {
      Net.Calculate(input, values);

      var lidx   = Net.LayerCount - 1;
      var result = values[lidx];
      var len    = result.GetLength(0);
      var output = new double[len];
      for (int j=0; j<len; j++) output[j] = result[j][0, 0];

      var expect = m_ExpectedOutputs[cls];
      var llayer = Net[lidx];

      for (int p=0; p<llayer.OutputDepth; p++)
      {
        var ej = m_LossFunction.Derivative(p, output, expect);
        var value = result[p][0, 0];
        var deriv = (llayer.ActivationFunction != null) ? llayer.ActivationFunction.DerivativeFromValue(value) : 1;
        errors[lidx][p][0, 0] = ej * deriv / m_BatchSize;
      }

      var loss = m_LossFunction.Value(output, expect) / m_BatchSize;
      if (m_Regularizator != null)
        loss += (m_Regularizator.Value(Net.Weights) / m_BatchSize);

      return loss;
    }

    #endregion
  }
}
