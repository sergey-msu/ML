using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ML.Core;
using ML.Contracts;
using ML.Core.Mathematics;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Represents simple Rosenblatt perceptron with a single hidden layer and scalar {0, 1} output
  /// </summary>
  public class PerceptronAlgorithm : NeuralNetworkAlgorithmBase
  {
    #region Inner

    public struct SAConnection
    {
      public int SIndex { get; set; }
      public int AIndex { get; set; }
      public int Value  { get; set; }
    }

    #endregion

    private readonly int m_HiddenNeuronCount;
    private readonly int m_InputDimension;
    private readonly double m_MinShift;
    private readonly double m_MaxShift;
    private readonly List<SAConnection> m_SAConnections;
    private readonly RandomGenerator m_RandomGenerator;

    public PerceptronAlgorithm(ClassifiedSample classifiedSample,
                               int inputDimension,
                               int hiddenNeuronCount,
                               double minShift,
                               double maxShift,
                               List<SAConnection> saConnections = null)
      : base(classifiedSample)
    {
      if (hiddenNeuronCount <= 0)
        throw new MLException("PerceptronAlgorithm.ctor(hiddenNeuronCount<=0)");
      if (inputDimension <= 0)
        throw new MLException("PerceptronAlgorithm.ctor(inputDimension<=0)");

      m_MinShift = minShift;
      m_MaxShift = maxShift;
      m_InputDimension = inputDimension;
      m_HiddenNeuronCount = hiddenNeuronCount;
      m_SAConnections = saConnections;

      m_RandomGenerator = new RandomGenerator();

      createPerceptronNetwork();
    }

    public override string ID { get { return "RPER"; } }
    public override string Name { get { return "Single-Hidden Layer Rosenblatt Perceptron"; } }


    public override Class Classify(Point x)
    {
      if (x.Dimension != m_InputDimension)
        throw new MLException("Input dimention differs from the declared");

      var output = Network.Calculate(x);
      if (output==null || output.Length != 1)
        throw new MLException("Perceptron predefined structure was corrupted from outside");

      var cls1 = Classes.FirstOrDefault(c => (int)c.Value.Value == 0);
      var cls2 = Classes.FirstOrDefault(c => (int)c.Value.Value == 1);

      var cls = Class.None;
      if ((int)output[0]==1) cls = cls1.Value;
      else if ((int)output[0]==-1) cls = cls2.Value;

      return cls;
    }

    public void Train_ErrorCorrection(int epochCount)
    {
      for (int i=0; i<epochCount; i++)
      {
        foreach (var pdata in TrainingSample)
        {
          var result = Network.Layers[0].Calculate(pdata.Key);
          if (result==null || result.Length != m_HiddenNeuronCount)
            throw new MLException("Perceptron predefined structure was corrupted from outside");

          var modifier = (int)pdata.Value.Value == 0 ? 1 : -1;
          for (int j=0; j<result.Length; j++)
          {
            var value = result[j];
            if ((int)value == 1) Network.TrySetWeight(1, j, 0, modifier, true);
          }
        }
      }
    }

    #region .pvt

    private void createPerceptronNetwork()
    {
      var associativeLayer = Network.CreateLayer();

      for (int i=0; i<m_HiddenNeuronCount; i++)
      {
        var neuron = associativeLayer.CreateNeuron();
        var shift = m_RandomGenerator.GenerateUniform(m_MinShift, m_MaxShift);
        neuron.ActivationFunction = Registry.ActivationFunctions.ShiftedStep(shift);

        if (m_SAConnections==null)
        {
          for (int j=0; j<m_InputDimension; j++)
            neuron[j] = 1.0D;
        }
        else
        {
          var connections = m_SAConnections.Where(c => c.AIndex==i);
          foreach (var connection in connections)
            neuron[connection.SIndex] = connection.Value;
        }
      }

      var reactLayer = Network.CreateLayer();
      reactLayer.ActivationFunction = Registry.ActivationFunctions.Sign;
      var output = reactLayer.CreateNeuron();
      for (int i=0; i<m_HiddenNeuronCount; i++)
        output[i] = 0.0D;
    }

    #endregion
  }
}
