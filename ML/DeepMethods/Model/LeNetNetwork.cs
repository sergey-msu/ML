using System;
using System.Collections.Generic;
using ML.Core.ComputingNetworks;
using ML.NeuralMethods.Model;
using ML.Core;

namespace ML.DeepMethods.Model
{
  /// <summary>
  /// Represents LeNet convolutional network - a set of convolutional layers with pooling
  /// followed by fully-connected neural network layers
  /// </summary>
  public class LeNetNetwork : ComputingNode<double[][,], double[]>
  {
    #region Fields

    private ConvolutionalNetwork m_ConvolutionalTier;
    private NeuralNetwork m_FullyConnectedTier;
    private int m_InputDim;

    #endregion

    #region .ctor

    public LeNetNetwork(int inputDim, int inputSize, int fullConnectedInputDim)
    {
      if (inputDim <= 0)
        throw new MLException("LeNetNetwork.ctor(inputDim<=0)");
      if (inputSize <= 0)
        throw new MLException("LeNetNetwork.ctor(inputSize<=0)");
      if (fullConnectedInputDim <= 0)
        throw new MLException("LeNetNetwork.ctor(fullConnectedInputDim<=0)");

      m_InputDim = inputDim;
      m_ConvolutionalTier  = new ConvolutionalNetwork(inputDim, inputSize);
      m_FullyConnectedTier = new NeuralNetwork(fullConnectedInputDim);
    }

    #endregion

    #region Properties

    public override int ParamCount { get { return 0; } }

    public ConvolutionalNetwork ConvolutionalTier { get { return m_ConvolutionalTier; } }

    public NeuralNetwork FullyConnectedTier { get { return m_FullyConnectedTier; } }

    /// <summary>
    /// Total count of network layers (convolutional + hidden + output)
    /// </summary>
    public int LayerCount { get { return m_ConvolutionalTier.LayerCount+m_FullyConnectedTier.LayerCount; } }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim { get { return m_InputDim; } }

    #endregion

    #region Public

    public override double[] Calculate(double[][,] input)
    {
      var featureMaps = m_ConvolutionalTier.Calculate(input);

      var lastConvolution = m_ConvolutionalTier[m_ConvolutionalTier.LayerCount-1];
      var convLen = lastConvolution.NeuronCount;
      var convSize = lastConvolution.OutputSize;
      var len = convLen * convSize * convSize;
      var featureInput = new double[len];

      int idx = 0;
      for (int h=0; h<convLen; h++)
      {
        var fm = featureMaps[h];
        for (int i=0; i<convSize; i++)
        for (int j=0; j<convSize; j++)
          featureInput[idx++] = fm[i, j];
      }

      return m_FullyConnectedTier.Calculate(featureInput);
    }

    #endregion

    #region Protected

    protected override double DoGetParam(int idx)
    {
      throw new NotSupportedException();
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
      throw new NotSupportedException();
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
      throw new NotSupportedException();
    }

    #endregion
  }
}
