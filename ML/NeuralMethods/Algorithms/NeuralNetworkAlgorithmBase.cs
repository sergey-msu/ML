using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Core.Mathematics;
using ML.NeuralMethods.Model;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Feedforward Neural Network machine learning algorithm
  /// </summary>
  public abstract class NeuralNetworkAlgorithmBase : AlgorithmBase
  {
    private readonly RandomGenerator m_Random;
    private NeuralNetwork m_Result;

    protected NeuralNetworkAlgorithmBase(ClassifiedSample classifiedSample)
      : base(classifiedSample)
    {
      m_Random = new RandomGenerator();
    }

    /// <summary>
    /// The result of the algorithm
    /// </summary>
    public NeuralNetwork Result { get { return m_Result; } }

    /// <summary>
    /// If true randomize in [0, 1] all neuron weights befor begin training
    /// </summary>
    public bool RandomizeInitialWeights { get; set; }

    /// <summary>
    /// Maps object to corresponding class
    /// </summary>
    public override Class Classify(Point x)
    {
      var result = m_Result.Calculate(x);
      var len = result.Length;
      Class cls;

      if (len==1)
      {
        cls = Classes.FirstOrDefault(c => (int)c.Value.Value == (int)result[0]).Value ?? Class.None;
      }
      else
      {
        int idx;
        double max;
        MathUtils.CalcMax(result, out idx, out max);

        cls = Classes.FirstOrDefault(c => (int)c.Value.Value == idx+1).Value  ?? Class.None;
      }

      return cls;
    }

    /// <summary>
    /// Prepares all the data to algorithm training
    /// </summary>
    public void Build()
    {
      m_Result = DoBuild();

      if (RandomizeInitialWeights) randomizeWeights(m_Result);
    }

    /// <summary>
    /// Teaches algorithm, produces Network output
    /// </summary>
    public void Train()
    {
      DoTrain();
    }

    protected abstract NeuralNetwork DoBuild();
    protected abstract void DoTrain();

    #region .pvt

    private void randomizeWeights(NeuralNetwork net)
    {
      foreach (var layer in net.SubNodes)
      foreach (var neuron in layer.SubNodes)
      {
        for (int i=0; i<neuron.ParamCount; i++)
          neuron[i] = m_Random.GenerateUniform(0, 1);
      }
    }

    #endregion
  }
}
