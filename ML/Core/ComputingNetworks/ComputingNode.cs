using System;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents abstract computing node
  ///
  /// TIn -> TOut
  ///
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  public abstract class ComputingNode<TIn, TOut> : ComputingNode, IComputingNode<TIn, TOut>
  {
    public override object Calculate(object input)
    {
      return Calculate((TIn)input);
    }

    public abstract TOut Calculate(TIn input);
  }


  /// <summary>
  /// Represents abstract computing node
  ///
  /// TIn -> TOut
  ///
  /// </summary>
  public abstract class ComputingNode : IComputingNode
  {
    private ParamIdx m_ParIdx;

    protected ComputingNode()
    {
    }

    /// <summary>
    /// Returns number of layer parameters
    /// </summary>
    public abstract int ParamCount { get; }

    /// <summary>
    /// Performs calculations and returns strong typed result
    /// </summary>
    public abstract object Calculate(object input);

    /// <summary>
    /// Builds layer before use
    /// </summary>
    public void Build()
    {
      DoBuild();
      BuildIndex(0);
    }

    /// <summary>
    /// Builds layer before use
    /// </summary>
    public virtual void DoBuild() { }

    /// <summary>
    /// Builds fast search index
    /// </summary>
    /// <param name="startIdx">Index start value</param>
    /// <returns>End index</returns>
    internal virtual int BuildIndex(int startIdx)
    {
      var endIdx = startIdx + ParamCount;
      m_ParIdx = new ParamIdx(startIdx, endIdx);
      return endIdx;
    }

    /// <summary>
    /// Tries to return parameter value at some position
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method COMPLETELY or not override at all
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public virtual bool TryGetParam(int idx, out double value)
    {
      //if (!m_ParIdx.CheckEnd(idx))
      //{
      //  value = 0;
      //  return false;
      //}
      //
      //ComputingNode subnode;
      //int subidx;
      //var res = TryGetSubnodeByParamIndex(idx, out subnode, out subidx);
      //if (res)
      //{
      //  value = subnode.DoGetParam(subidx);
      //  return true;
      //}
      //
      //value = 0;
      //return false;

      if (!m_ParIdx.CheckEnd(idx))
      {
        value = 0;
        return false;
      }

      value = DoGetParam(idx-m_ParIdx.Start);
      return true;
    }

    /// <summary>
    /// Tries to set parameter value at some position
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method COMPLETELY or not override at all
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <param name="isDelta">Is the values are exact or just delta to existing one</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public virtual bool TrySetParam(int idx, double value, bool isDelta)
    {
      //if (!m_ParIdx.CheckEnd(idx))
      //  return false;
      //
      //ComputingNode subnode;
      //int subidx;
      //var res = TryGetSubnodeByParamIndex(idx, out subnode, out subidx);
      //if (res)
      //{
      //  subnode.DoSetParam(subidx, value, isDelta);
      //  return true;
      //}
      //
      //return false;

      if (!m_ParIdx.CheckEnd(idx))
        return false;

      DoSetParam(idx-m_ParIdx.Start, value, isDelta);
      return true;
    }

    /// <summary>
    /// Updates parameters of the network and passes it down to all sublayers
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method COMPLETELY or not override at all
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    public virtual bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (pars == null || pars.Length < cursor+ParamCount) return false;

      DoUpdateParams(pars, isDelta, cursor);
      cursor += ParamCount;

      return true;
    }

    protected abstract double DoGetParam(int idx);
    protected abstract void DoSetParam(int idx, double value, bool isDelta);
    protected abstract void DoUpdateParams(double[] pars, bool isDelta, int cursor);
  }
}
