using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents hidden computing layer -
  /// non-terminating layer that calculates some result and passes it further in the layer chain
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Direct result (not final) type</typeparam>
  public abstract class HiddenLayer<TIn, TOut> : IHiddenLayer<TIn, TOut>
  {
    private ParamIdx m_ParIdx;
    private IComputingLayer<TOut> m_NextLayer;

    public HiddenLayer()
    {
    }

    /// <summary>
    /// Returns number of layer parameters
    /// </summary>
    public abstract int ParamsCount { get; }

    /// <summary>
    /// Next layer in layer chain
    /// </summary>
    public IComputingLayer<TOut> NextLayer { get { return m_NextLayer; } }

    /// <summary>
    /// Adds next (it terms of linked list paradigm) layer
    /// </summary>
    public void AddNextLayer(IComputingLayer<TOut> layer)
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
    /// Builds layer before use (build search index etc)
    /// </summary>
    public virtual void Build(bool buildIndex = true)
    {
      if (m_NextLayer==null)
        throw new MLException("Next layer has not been set");

      if (buildIndex) BuildIndex(0);
      m_NextLayer.Build(false);
    }

    /// <summary>
    /// Builds fast search index
    /// </summary>
    /// <param name="startIdx">Index start value</param>
    /// <returns>End index</returns>
    public int BuildIndex(int startIdx)
    {
      if (m_NextLayer==null)
        throw new MLException("Next layer has not been set");

      var selfIdx = startIdx+ParamsCount;
      var endIdx = m_NextLayer.BuildIndex(selfIdx);
      m_ParIdx = new ParamIdx(startIdx, selfIdx, endIdx);

      return endIdx;
    }

    /// <summary>
    /// Tries to return parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public bool TryGetParam(int idx, out double value)
    {
      if (m_ParIdx.CheckEnd(idx))
      {
        return m_NextLayer.TryGetParam(idx, out value);
      }

      if (m_ParIdx.CheckSelf(idx))
      {
        value = DoGetParam(idx-m_ParIdx.Start);
        return true;
      }

      value = 0;
      return false;
    }

    /// <summary>
    /// Tries to set parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <param name="isDelta">Is the values are exact or just delta to existing one</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public bool TrySetParam(int idx, double value, bool isDelta)
    {
      if (m_ParIdx.CheckEnd(idx))
      {
        return m_NextLayer.TrySetParam(idx, value, isDelta);
      }

      if (m_ParIdx.CheckSelf(idx))
      {
        DoSetParam(idx-m_ParIdx.Start, value, isDelta);
        return true;
      }

      return false;
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
      if (pars == null || pars.Length < cursor+ParamsCount)
        return false;

      DoUpdateParams(pars, isDelta, cursor);
      m_ParIdx.MoveCursor(ref cursor);

      return m_NextLayer.TryUpdateParams(pars, isDelta, ref cursor);
    }

    protected abstract TOut DoCalculate(TIn input);
    protected abstract double DoGetParam(int idx);
    protected abstract void DoSetParam(int idx, double value, bool isDelta);
    protected abstract void DoUpdateParams(double[] pars, bool isDelta, int cursor);
  }
}
