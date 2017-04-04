using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Core.Mathematics;
using ML.NeuralMethods.Models;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Feedforward Neural Network machine learning algorithm
  /// </summary>
  public abstract class NeuralNetworkAlgorithmBase : AlgorithmBase<double[]>
  {
    private NeuralNetwork m_Result;

    protected NeuralNetworkAlgorithmBase(ClassifiedSample<double[]> classifiedSample, NeuralNetwork net)
      : base(classifiedSample)
    {
      if (net==null)
        throw new MLException("Network can not be null");

      m_Result = net;
      net.IsTraining = true;
    }

    /// <summary>
    /// The result of the algorithm
    /// </summary>
    public NeuralNetwork Result { get { return m_Result; } }

    /// <summary>
    /// Maps object to corresponding class
    /// </summary>
    public override Class Classify(double[] x)
    {
      var result = m_Result.Calculate(x);
      var len = result.Length;
      Class cls;

      int idx;
      double max;
      MathUtils.CalcMax(result, out idx, out max);

      cls = Classes.FirstOrDefault(c => (int)c.Value.Value == idx).Value  ?? Class.None;

      return cls;
    }

    public override IEnumerable<ErrorInfo> GetErrors(ClassifiedSample<double[]> classifiedSample)
    {
      var isTraining = m_Result.IsTraining;
      m_Result.IsTraining = false;
      var errors = base.GetErrors(classifiedSample);
      m_Result.IsTraining = isTraining;

      return errors;
    }

    /// <summary>
    /// Teaches algorithm, produces Result output
    /// </summary>
    public void Train()
    {
      DoTrain();
      m_Result.IsTraining = false;
    }

    protected abstract void DoTrain();
  }
}
