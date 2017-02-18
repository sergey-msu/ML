using System;
using System.Collections.Generic;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents output computing layer - terminating layer that calculates final result in the layer chain
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Result type</typeparam>
  public abstract class OutputLayer<TIn, TOut> : IOutputLayer<TIn, TOut>
  {
    private ParamIdx m_ParIdx;

    public OutputLayer()
    {
    }

    /// <summary>
    /// Returns number of layer parameters
    /// </summary>
    public abstract int ParamsCount { get; }

    /// <summary>
    /// Performs calculations and returns strong typed result
    /// </summary>
    public abstract TOut Calculate(TIn input);

    /// <summary>
    /// Builds layer before use (build search index etc)
    /// </summary>
    public virtual void Build(bool buildIndex = true)
    {
      if (buildIndex) BuildIndex(0);
    }

    /// <summary>
    /// Builds fast search index
    /// </summary>
    /// <param name="startIdx">Index start value</param>
    /// <returns>End index</returns>
    public int BuildIndex(int startIdx)
    {
      var endIdx = startIdx + ParamsCount;
      m_ParIdx = new ParamIdx(startIdx, endIdx, endIdx);
      return m_ParIdx.End;
    }

    /// <summary>
    /// Passes input through linked list of sublayers and returns the result
    /// </summary>
    object IComputingLayer<TIn>.Calculate(TIn input)
    {
      return Calculate(input);
    }

    /// <summary>
    /// Tries to return parameter value at some position
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public bool TryGetParam(int idx, out double value)
    {
      if (m_ParIdx.CheckSelf(idx)) // self params count = total pam count for terminating layer
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
      if (m_ParIdx.CheckSelf(idx)) // self params count = total pam count for terminating layer
      {
        DoSetParam(idx-m_ParIdx.Start, value, isDelta);
        return true;
      }

      return false;
    }

    /// <summary>
    /// Updates parameters of the network and passes it down to all sublayers
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    public bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (pars == null || pars.Length < cursor+ParamsCount) return false;

      DoUpdateParams(pars, isDelta, cursor);
      m_ParIdx.MoveCursor(ref cursor);

      return true;
    }

    protected abstract double DoGetParam(int idx);
    protected abstract void DoSetParam(int idx, double value, bool isDelta);
    protected abstract void DoUpdateParams(double[] pars, bool isDelta, int cursor);
  }
}
