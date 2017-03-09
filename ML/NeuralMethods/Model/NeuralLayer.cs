using System;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Model
{
  /// <summary>
  /// Represents artificial neural layer as a list of neurons
  /// </summary>
  public class NeuralLayer : CompositeNode<double[], double, Neuron>
  {
    private readonly NeuralNetwork m_Network;
    private IFunction m_ActivationFunction;
    private int m_InputDim;

    public NeuralLayer(int inputDim)
    {
      if (inputDim <= 0)
        throw new MLException("NeuralNetwork.ctor(inputDim<=0)");

      m_InputDim = inputDim;
    }

    public NeuralLayer(NeuralNetwork network, int inputDim)
      : this(inputDim)
    {
      if (network == null)
        throw new MLException("NeuronLayer.ctor(network=null)");

      m_Network = network;
    }


    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim
    {
      get { return m_InputDim; }
      internal set
      {
        m_InputDim=value;

        for (int i=0; i<NeuronCount; i++)
          this[i].InputDim = value;
      }
    }

    /// <summary>
    /// Total count of neurons
    /// </summary>
    public int NeuronCount { get { return SubNodes.Length; } }

    /// <summary>
    /// In true, norms output vector to the summ of absolute values of its components (give more 'probabalistic' meaning to the result)
    /// </summary>
    public bool NormOutput { get; set; }

    /// <summary>
    /// Neural network that the layer belongs to
    /// </summary>
    public NeuralNetwork Network { get { return m_Network; } }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for layer neurons
    /// </summary>
    public Neuron this[int idx] { get { return SubNodes[idx]; } }


    /// <summary>
    /// Creates new flat neuron and adds it into its nodes collection
    /// </summary>
    /// <returns></returns>
    public Neuron CreateNeuron<TNeuron>()
      where TNeuron : Neuron
    {
      var neuron = (TNeuron)Activator.CreateInstance(typeof(TNeuron), this);
      this.AddSubNode(neuron);
      return neuron;
    }

    /// <summary>
    /// Add existing neuron in the end of the layer
    /// </summary>
    public void AddNeuron(Neuron neuron)
    {
      if (neuron==null)
        throw new MLException("Neuron can not be null");
      if (neuron.InputDim != this.InputDim)
        throw new MLException("Neuron input dimension differs with layer's one");

      this.AddSubNode(neuron);
    }

    /// <summary>
    /// Randomizes layer's neuron weights
    /// </summary>
    public void Randomize(int seed=0)
    {
      foreach (var neuron in this.SubNodes)
        neuron.Randomize(seed);
    }

    /// <summary>
    /// Calculates result array produced by layer
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double[] Calculate(double[] input)
    {
      if (InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      var result = base.Calculate(input);

      if (NormOutput)
      {
        var sum = 0.0D;
        var len = result.Length;
        for (int i = 0; i < len; i++)
          sum += Math.Abs(result[i]);

        if (sum > 0)
        {
          for (int i = 0; i < len; i++)
            result[i] /= sum;
        }
      }

      return result;
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      if (ActivationFunction==null)
        ActivationFunction = (Network != null ? Network.ActivationFunction : null) ?? Registry.ActivationFunctions.Identity;

      base.DoBuild();
    }
  }
}
