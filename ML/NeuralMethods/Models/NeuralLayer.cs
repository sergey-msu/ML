using System;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Models
{
  /// <summary>
  /// Represents artificial neural layer as a list of neurons
  /// </summary>
  public class NeuralLayer : CompositeNode<double[], double, Neuron>
  {
    #region Fields

    private IActivationFunction m_ActivationFunction;
    private int m_InputDim;

    #endregion

    #region .ctor

    public NeuralLayer(int inputDim)
    {
      if (inputDim <= 0)
        throw new MLException("NeuralLayer.ctor(inputDim<=0)");

      m_InputDim = inputDim;
    }

    #endregion

    #region Properties

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim { get { return m_InputDim; } }

    /// <summary>
    /// Total count of abstract neurons (i.e. neurons in NN, feature maps in CNN etc.)
    /// </summary>
    public int NeuronCount { get { return SubNodes.Length; } }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IActivationFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for layer abstract neurons (i.e. neurons in NN, feature maps in CNN etc.)
    /// </summary>
    public Neuron this[int idx] { get { return SubNodes[idx]; } }

    #endregion

    #region Public

    /// <summary>
    /// Add existing abstract neurons (i.e. neurons in NN, feature maps in CNN etc.) in the end of the layer
    /// </summary>
    public virtual void AddNeuron(Neuron neuron)
    {
      if (neuron==null)
        throw new MLException("Neuron can not be null");
      if (neuron.InputDim != m_InputDim)
        throw new MLException("Neuron input dimension differs with layer's one");

      this.AddSubNode(neuron);
    }

    /// <summary>
    /// Randomizes layer neurons parameters (weights for NNs)
    /// </summary>
    public virtual void RandomizeParameters(int seed=0)
    {
      foreach (var neuron in this.SubNodes)
        neuron.RandomizeParameters(seed);
    }

    /// <summary>
    /// Calculates result array produced by layer
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double[] Calculate(double[] input)
    {
      if (InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      return base.Calculate(input);
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      foreach (var neuron in this.SubNodes)
        neuron.ActivationFunction = neuron.ActivationFunction ?? ActivationFunction;

      base.DoBuild();
    }

    #endregion
  }
}
