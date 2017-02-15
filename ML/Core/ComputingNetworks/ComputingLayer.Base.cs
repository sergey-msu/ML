using System;
using System.Collections.Generic;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents output computing layer - terminating layer that calculates final result in the layer chain
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Result type</typeparam>
  public abstract class OutputLayer<TIn, TOut> : IComputingLayer<TIn>
  {
    public OutputLayer()
    {
    }

    /// <summary>
    /// Performs calculations and returns strong typed result
    /// </summary>
    public abstract TOut Calculate(TIn input);

    /// <summary>
    /// Passes input through linked list of sublayers and returns the result
    /// </summary>
    object IComputingLayer<TIn>.Calculate(TIn input)
    {
      return Calculate(input);
    }

    /// <summary>
    /// Updates parameters of the network and passes it down to all sublayers
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    public abstract bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor);

    /// <summary>
    /// Tries to set parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <param name="isDelta">Is the values are exact or just delta to existing one</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public abstract bool TrySetParam(ref int idx, double value, bool isDelta);

    /// <summary>
    /// Tries to return parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public abstract bool TryGetParam(ref int idx, out double value);
  }

  /// <summary>
  /// Represents hidden computing layer -
  /// non-terminating layer that calculates some result and passes it further in the layer chain
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Direct result (not final) type</typeparam>
  public abstract class HiddenLayer<TIn, TOut> : IComputingLayer<TIn>
  {
    private IComputingLayer<TOut> m_NextLayer;

    public HiddenLayer()
    {
    }

    /// <summary>
    /// Next layer in layer chain
    /// </summary>
    public IComputingLayer<TOut> NextLayer { get { return m_NextLayer; } }

    public void AddNext(IComputingLayer<TOut> layer)
    {
      if (layer==null)
        throw new MLException("Next layer can not be null");

      m_NextLayer = layer;
    }

    /// <summary>
    /// Calculates final calculation result.
    /// It may differ (for non-terminating layers) from direct layer rcalculation result:
    /// L = [L1 -> [L2 -> L3]]
    /// L(x) = L3(L2(L1(x))) - the final result, not L1(x)
    /// </summary>
    public object Calculate(TIn input)
    {
      if (m_NextLayer==null)
        throw new MLException("Next layer has not been set");

      var result = DoCalculate(input);
      return m_NextLayer.Calculate(result);
    }

    /// <summary>
    /// Tries to update parameters of the network and passes it down to all sublayers
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    public bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (pars == null || pars.Length <= cursor || m_NextLayer == null)
        return false;

      var success = DoUpdateParams(pars, isDelta, ref cursor);
      if (!success) return false;

      return m_NextLayer.TryUpdateParams(pars, isDelta, ref cursor);
    }

    /// <summary>
    /// Tries to return parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public bool TryGetParam(ref int idx, out double value)
    {
      var success = DoGetParam(ref idx, out value);
      if (success) return true;

      value = 0;
      return m_NextLayer.TryGetParam(ref idx, out value);
    }

    /// <summary>
    /// Tries to set parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <param name="isDelta">Is the values are exact or just delta to existing one</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public bool TrySetParam(ref int idx, double value, bool isDelta)
    {
      var success = DoSetParam(ref idx, value, isDelta);
      if (success) return true;

      return m_NextLayer.TrySetParam(ref idx, value, isDelta);
    }


    protected abstract bool DoUpdateParams(double[] pars, bool isDelta, ref int cursor);
    protected abstract bool DoGetParam(ref int idx, out double value);
    protected abstract bool DoSetParam(ref int idx, double value, bool isDelta);
    protected abstract TOut DoCalculate(TIn input);
  }

  /// <summary>
  /// Represents composite computational layer -
  /// a hidden layer that aggregates another layers and merges its outputs
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Direct result (not final) type</typeparam>
  public abstract class CompositeLayer<TIn, TOut> : HiddenLayer<TIn, TOut>
  {
    public List<IComputingLayer<TIn>> m_SubLayers = new List<IComputingLayer<TIn>>();

    public void AddSubLayer(IComputingLayer<TIn> subLayer)
    {
      if (subLayer==null)
        throw new MLException("Sublayer can not be null");

      m_SubLayers.Add(subLayer);
    }

    protected override TOut DoCalculate(TIn input)
    {
      var len = m_SubLayers.Count;
      var results = new TOut[len];
      for (int i = 0; i < len; i++)
        results[i] = (TOut)m_SubLayers[i].Calculate(input);

      return MergeResults(results);
    }

    protected abstract TOut MergeResults(TOut[] results);
  }
}
