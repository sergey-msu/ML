﻿using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.DeepMethods.Model;
using ML.Contracts;

namespace ML.DeepMethods.Algorithms
{
  /// <summary>
  /// Multi-layer convolutional neural network training algorithm.
  /// Using Backpropagation principle as a core.
  /// </summary>
  public class BackpropAlgorithm : ConvolutionalNetworkAlgorithmBase
  {
    #region Inner

    public enum StopCriteria
    {
      FullLoop = 0,
      ErrorFunc = 1,
      StepMin = 2,
      QFunc = 3
    }

    #endregion

    #region CONST

    public const int DFT_EPOCH_COUNT = 1;
    public const double DFT_LEARNING_RATE = 0.1D;
    public const double DFT_Q_LAMBDA = 0.9D;
    public const StopCriteria DTF_STOP_CRITERIA = StopCriteria.FullLoop;
    public static readonly IFunction DFT_ACTIVATION_FUNCTION = Registry.ActivationFunctions.Identity;

    #endregion

    #region Fields

    private Dictionary<Class, double[]> m_ExpectedOutputs;
    private int m_EpochLength;

    private StopCriteria m_Stop;
    private int m_InputDepth;
    private int m_InputSize;
    private int m_OutputDepth;
    private int m_EpochCount;
    private double m_LearningRate;

    private double m_IterErrorValue;
    private double m_PrevErrorValue;
    private double m_ErrorValue;
    private double m_ErrorDelta;
    private double m_ErrorStopDelta;

    private double m_Step2;
    private double m_StepStopValue;

    private double m_QLambda;
    private double m_PrevQValue;
    private double m_QValue;
    private double m_QDelta;
    private double m_QStopDelta;

    #endregion

    #region .ctor

    public BackpropAlgorithm(ClassifiedSample<double[,,]> classifiedSample, ConvolutionalNetwork net)
      : base(classifiedSample, net)
    {
      init();
    }

    #endregion

    #region Events

    public event EventHandler EpochEndedEvent;

    #endregion

    #region Properties

    public override string ID { get { return "CNN_BP"; } }
    public override string Name { get { return "Convolutional Neural Network with Backpropagation"; } }

    public int InputDepth { get { return m_InputDepth; } }
    public int InputSize { get { return m_InputSize; } }
    public int OutputDepth { get { return m_OutputDepth; } }
    public double IterErrorValue { get { return m_IterErrorValue; } }
    public double ErrorValue { get { return m_ErrorValue; } }
    public double ErrorDelta { get { return m_ErrorDelta; } }
    public double Step2 { get { return m_Step2; } }
    public double QValue { get { return m_QValue; } }
    public double QDelta { get { return m_QDelta; } }

    public int EpochCount
    {
      get { return m_EpochCount; }
      set
      {
        if (value <= 0)
          throw new MLException("Epoch count must be positive");
        m_EpochCount = value;
      }
    }

    public double LearningRate
    {
      get { return m_LearningRate; }
      set
      {
        if (LearningRate <= 0)
          throw new MLException("Learning rate must be positive");
        m_LearningRate = value;
      }
    }

    public StopCriteria Stop
    {
      get { return m_Stop; }
      set { m_Stop = value; }
    }

    public double ErrorStopDelta
    {
      get { return m_ErrorStopDelta; }
      set
      {
        if (ErrorStopDelta < 0)
          throw new MLException("Error Stop Delta must be non-negative");
        m_ErrorStopDelta = value;
      }
    }

    public double StepStopValue
    {
      get { return m_StepStopValue; }
      set
      {
        if (StepStopValue < 0)
          throw new MLException("Step Stop Value must be non-negative");
        m_StepStopValue = value;
      }
    }

    public double QLambda
    {
      get { return m_QLambda; }
      set
      {
        if (QLambda < 0 || QLambda > 1)
          throw new MLException("Lambda for Q-stop criteria must be in [0, 1] interval");
        m_QLambda = value;
      }
    }

    public double QStopDelta
    {
      get { return m_QStopDelta; }
      set
      {
        if (QStopDelta < 0)
          throw new MLException("Q Stop Delta must be non-negative");
        m_QStopDelta = value;
      }
    }

    #endregion

    #region Public

    public void RunEpoch()
    {
      runEpoch(Result);
    }

    public void RunIteration(double[,,] data, Class cls)
    {
      runIteration(Result, data, cls);
    }

    #endregion

    public override void Train()
    {
      doTrain(Result);
    }

    #region .pvt

    private void init()
    {
      m_EpochCount = DFT_EPOCH_COUNT;
      m_LearningRate = DFT_LEARNING_RATE;
      m_Stop = DTF_STOP_CRITERIA;
      m_QLambda = DFT_Q_LAMBDA;
      m_EpochLength = TrainingSample.Count;
      m_InputDepth = Result.InputDepth;
      m_InputSize = Result.InputSize;
      m_OutputDepth = Result[Result.LayerCount - 1].OutputDepth;

      m_ExpectedOutputs = new Dictionary<Class, double[]>();
      var count = Classes.Count;
      if (count != OutputDepth)
        throw new MLException("Number of classes must be equal to dimension of output vector");

      for (int i=0; i<count; i++)
      {
        var cls = Classes.FirstOrDefault(p => (int)p.Value.Value == i).Value;
        if (cls == null)
          throw new MLException(string.Format("There is no class with value {0}. It is neccessary to have full set of classes with values from 0 to {1}", i, count));

        var output = new double[count];
        output[i] = 1.0D;
        m_ExpectedOutputs[cls] = output;
      }
    }

    private void doTrain(ConvolutionalNetwork net)
    {
      for (int epoch = 0; epoch < m_EpochCount; epoch++)
      {
        runEpoch(net);
        if (checkStopCriteria()) break;
      }
    }

    private void runEpoch(ConvolutionalNetwork net)
    {
      foreach (var pdata in TrainingSample)
      {
        runIteration(net, pdata.Key, pdata.Value);
      }

      // update epoch stats
      m_PrevErrorValue = m_ErrorValue;
      m_ErrorValue = m_IterErrorValue / TrainingSample.Count;
      m_ErrorDelta = m_ErrorValue - m_PrevErrorValue;
      m_IterErrorValue = 0.0D;

      if (EpochEndedEvent != null) EpochEndedEvent(this, EventArgs.Empty);
    }

    private void runIteration(ConvolutionalNetwork net, double[,,] input, Class cls)
    {
      // forward calculation
      var serr2 = feedForward(net, input, cls);

      // error backpropagation
      var lcount = net.LayerCount;
      for (int i=lcount-1; i>=0; i--)
      {
        var layer = net[i];
        if (layer is ConvolutionalLayer)
          feedBackward(net, (ConvolutionalLayer)layer, i, input);
        else if (layer is MaxPoolingLayer)
          feedBackward(net, (MaxPoolingLayer)layer, i, input);
        else if (layer is AvgPoolingLayer)
          feedBackward(net, (MaxPoolingLayer)layer, i, input);
        else
          throw new MLException("Unknown layer type");
      }

      // update iter stats
      m_IterErrorValue += serr2/2;
      m_PrevQValue = m_QValue;
      m_QValue = (1 - m_QLambda) * m_QValue + m_QLambda * serr2 / 2;
      m_QDelta = m_QValue - m_PrevQValue;
    }

    private double feedForward(ConvolutionalNetwork net, double[,,] input, Class cls)
    {
      var serr2 = 0.0D;
      var output = net.Calculate(input);
      var expect = m_ExpectedOutputs[cls];
      var llayer = net[net.LayerCount - 1];

      for (int j=0; j<m_OutputDepth; j++)
      {
        var ej = output[j, 0, 0] - expect[j];
        llayer.Error[j, 0, 0] = ej * llayer.Derivative(j, 0, 0);
        serr2 += ej * ej;
      }

      return serr2;
    }

    private void feedBackward(ConvolutionalNetwork net, MaxPoolingLayer layer, int lidx, double[,,] input)
    {
      if (lidx <= 0) return;

      var size   = layer.OutputSize;
      var depth  = layer.OutputDepth;
      var player = net[lidx-1];
      var psize  = player.OutputSize;
      if (depth != player.OutputDepth)
        throw new MLException("Network architecture inconsistency: max pooling layer can not change input depth");

      Array.Clear(player.Error, 0, player.Error.Length);

      for (int q=0; q<depth; q++)
      for (int i=0; i<size;  i++)
      for (int j=0; j<size;  j++)
      {
        var xmaxIdx = layer.MaxIndexPositions[q, i, j, 0];
        var ymaxIdx = layer.MaxIndexPositions[q, i, j, 1];
        player.Error[q, ymaxIdx, xmaxIdx] += layer.Error[q, i, j] * player.Derivative(q, ymaxIdx, xmaxIdx);
      }
    }

    private void feedBackward(ConvolutionalNetwork net, AvgPoolingLayer layer, int lidx, double[,,] input)
    {
      throw new NotImplementedException(); // TODO
    }

    private void feedBackward(ConvolutionalNetwork net, ConvolutionalLayer layer, int lidx, double[,,] input)
    {
      var depth   = layer.OutputDepth;
      var size    = layer.OutputSize;
      var wsize   = layer.WindowSize;
      var padding = layer.Padding;
      var stride  = layer.Stride;
      var player  = (lidx > 0) ? net[lidx - 1] : null;
      var pdepth  = (lidx > 0) ? player.OutputDepth : m_InputDepth;
      var psize   = (lidx > 0) ? player.OutputSize  : m_InputSize;

      var sstep2 = 0.0D;

      // backpropagate "errors" to previous layer for future use
      if (lidx > 0)
        for (int p=0; p<pdepth; p++)
        for (int i=0; i<psize; i++)
        for (int j=0; j<psize; j++)
        {
          var gpij = 0.0D;

          for (int q=0; q<depth; q++)
          for (int k=0; k<size; k++)
          {
            var iidx = i+padding-k*stride;
            if (iidx >= wsize) continue;
            if (iidx < 0) break;

            for (int m=0; m<size; m++)
            {
              var jidx = j+padding-m*stride;
              if (jidx >= wsize) continue;
              if (jidx < 0) break;

              gpij += layer.Error[q, k, m] * layer.Kernel[q, p, iidx, jidx];
            }
          }

          player.Error[p, i, j] = gpij * player.Derivative(p, i, j);
        }

      // update weights
      for (int q=0; q<depth; q++)
      {
        var berr = 0.0D; // TODO: we assume that bias is shared among all cells in each output feature map
                         //       i.e. there is ONLY ONE bias value FOR EACH feature map
                         //       Is it correct? Should we have m_Biases[q,i,j] instaed of m_Biases[q] ???
                         // http://datascience.stackexchange.com/questions/17671 - my question

        for (int i=0; i<size; i++)
        for (int j=0; j<size; j++)
        {
          var dj = m_LearningRate * layer.Error[q, i, j];
          berr += dj;

          for (int p=0; p<pdepth; p++)
          {
            var value = (lidx == 0) ? input[p, i, j] : player.Value[p, i, j];
            var dwj = dj * value;
            layer.Kernel[q, p, i, j] -= dwj;
            sstep2 += dwj * dwj;
          }
        }

        layer.Biases[q] -= berr;
        sstep2 += berr*berr;
      }

      // update iter stats
      m_Step2 = sstep2;
    }

    private bool checkStopCriteria()
    {
      switch (Stop)
      {
        case StopCriteria.FullLoop: return false;
        case StopCriteria.ErrorFunc: return Math.Abs(m_ErrorDelta) < m_ErrorStopDelta;
        case StopCriteria.QFunc: return Math.Abs(m_QDelta) < m_QStopDelta;
        case StopCriteria.StepMin: return m_Step2 < StepStopValue;
        default: throw new MLException("Unknown stop citeria");
      }
    }

    #endregion
  }
}
