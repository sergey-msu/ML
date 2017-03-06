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
    private NeuralNetwork m_Result;

    protected NeuralNetworkAlgorithmBase(ClassifiedSample classifiedSample)
      : base(classifiedSample)
    {
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
    /// Seed for random generator if RandomizeInitialWeights is set to true
    /// </summary>
    public int RandomSeed { get; set; }

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

      if (RandomizeInitialWeights) m_Result.Randomize(RandomSeed);
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
  }
}
