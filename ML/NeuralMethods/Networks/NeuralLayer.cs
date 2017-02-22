using System;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;
using ML.NeuralMethods.Contracts;

namespace ML.NeuralMethods.Networks
{
  /// <summary>
  /// Represents artificial neural layer as a list of neurons
  /// </summary>
  public class NeuralLayer : CompositeNode<double[], double, Neuron>
  {
    private readonly INeuralNetwork m_Network;
    private IFunction m_ActivationFunction;

    public NeuralLayer()
    {
    }

    public NeuralLayer(INeuralNetwork network)
    {
      if (network == null)
        throw new MLException("NeuronLayer.ctor(network=null)");

      m_Network = network;
    }

    /// <summary>
    /// If true, adds artificial +1 input value in the and of input data array
    /// </summary>
    public bool UseBias { get; set; }

    /// <summary>
    /// In true, norms output vector to the summ of absolute values of its components (give more 'probabalistic' meaning to the result)
    /// </summary>
    public bool NormOutput { get; set; }

    /// <summary>
    /// Neural network that the layer belongs to
    /// </summary>
    public INeuralNetwork Network { get { return m_Network; } }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IFunction ActivationFunction
    {
      get
      {
        if (m_ActivationFunction != null)
          return m_ActivationFunction;

        return m_Network != null ?
               m_Network.ActivationFunction ?? Registry.ActivationFunctions.Identity :
               Registry.ActivationFunctions.Identity;
      }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Creates new neuron and adds it into its nodes collection
    /// </summary>
    /// <returns></returns>
    public Neuron CreateNeuron()
    {
      var neuron = new Neuron(this);
      this.AddSubNode(neuron);
      return neuron;
    }

    /// <summary>
    /// Calculates result array produced by layer
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double[] Calculate(double[] input)
    {
      var data = input;

      if (UseBias)
      {
        data = new double[input.Length + 1];
        Buffer.BlockCopy(input, 0, data, 0, input.Length);
        data[input.Length] = 1.0D;
      }

      var result = base.Calculate(data);

      if (NormOutput)
      {
        var sum = 0.0D;
        var len = result.Length;
        for (int i = 0; i < len; i++)
          sum += Math.Abs(result[i]);
        for (int i = 0; i < len; i++)
          result[i] /= sum;
      }

      return result;
    }
  }
}
