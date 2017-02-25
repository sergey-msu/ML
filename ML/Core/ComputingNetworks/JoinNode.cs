using System;
using System.Collections.Generic;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Represents 'stapling' node joined from two others
  ///
  /// TIn -> THidOut -> TOut
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="THidOut">Hidden (inner) input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  public class JoinNode<TIn, THidOut, TOut> : ComputingNode<TIn, TOut>
  {
    private ParamMultiIdx m_ParIdx;
    private ComputingNode<TIn, THidOut> m_InputNode;
    private ComputingNode<THidOut, TOut> m_OutputNode;

    public JoinNode()
    {
    }

    public override int ParamCount { get { return 0; } }
    public IComputingNode<TIn, THidOut> InputNode { get { return m_InputNode; } }
    public IComputingNode<THidOut, TOut> OutputNode { get { return m_OutputNode; } }

    public void SetInputNode(ComputingNode<TIn, THidOut> inputNode)
    {
      if (inputNode==null)
        throw new MLException("Node can not be null");

      m_InputNode = inputNode;
    }

    public void SetOutputNode(ComputingNode<THidOut, TOut> outputNode)
    {
      if (outputNode==null)
        throw new MLException("Node can not be null");

      m_OutputNode = outputNode;
    }

    public override TOut Calculate(TIn input)
    {
      var innerResult = m_InputNode.Calculate(input);
      return m_OutputNode.Calculate(innerResult);
    }

    /// <summary>
    /// Builds node before use
    /// </summary>
    public override void DoBuild()
    {
      if (m_InputNode==null)
        throw new MLException("Input node has not been set");
      if (m_OutputNode==null)
        throw new MLException("Output node has not been set");

      m_InputNode.DoBuild();
      m_OutputNode.DoBuild();
    }

    /// <summary>
    /// Builds fast search index
    /// </summary>
    /// <param name="startIdx">Index start value</param>
    /// <returns>Index end value</returns>
    internal override int BuildIndex(int startIdx)
    {
      if (m_InputNode==null)
        throw new MLException("Input node has not been set");
      if (m_OutputNode==null)
        throw new MLException("Output node has not been set");

      var idxs = new int[3];
      idxs[0] = startIdx;

      var endIdx = m_InputNode.BuildIndex(startIdx);
      idxs[1] = endIdx;

      endIdx = m_OutputNode.BuildIndex(endIdx);
      idxs[2] = endIdx;

      m_ParIdx = new ParamMultiIdx(idxs);

      return endIdx;
    }

    ///// <summary>
    ///// Tries to return subnode which owns parameter with specified index
    ///// WARNING: override this method carefully!
    ///// Do not use base. as base methods operate with base class index which may differ from exact class index
    ///// and so it will return wrong results. Override this method COMPLETELY or not override at all
    ///// </summary>
    ///// <param name="idx">Linear index of the parameter</param>
    ///// <param name="TNode">Subnode</param>
    ///// <param name="subidx">Internal index part within the subnode</param>
    ///// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    //public override bool TryGetSubnodeByParamIndex<TNode>(int idx, out TNode result, out int subidx)
    //{
    //  if (!m_ParIdx.CheckEnd(idx))
    //  {
    //    result = null;
    //    subidx = -1;
    //    return false;
    //  }
    //
    //  if (m_ParIdx.CheckIdx(idx, 1))
    //    return m_InputNode.TryGetSubnodeByParamIndex(idx, out result, out subidx);
    //
    //  return m_OutputNode.TryGetSubnodeByParamIndex(idx, out result, out subidx);
    //}

    /// <summary>
    /// Tries to return parameter value at some position
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method completely or not override
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public override bool TryGetParam(int idx, out double value)
    {
      if (!m_ParIdx.CheckEnd(idx))
      {
        value = 0;
        return false;
      }

      if (m_ParIdx.CheckIdx(idx, 1))
        return m_InputNode.TryGetParam(idx, out value);

      return m_OutputNode.TryGetParam(idx, out value);
    }

    /// <summary>
    /// Tries to set parameter value at some position
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method completely or not override
    /// </summary>
    /// <param name="idx">Linear index of the parameter</param>
    /// <param name="value">Parameter value</param>
    /// <param name="isDelta">Is the values are exact or just delta to existing one</param>
    /// <returns>True is operation succeeded, false otherwise (unexisted index etc.)</returns>
    public override bool TrySetParam(int idx, double value, bool isDelta)
    {
      if (!m_ParIdx.CheckEnd(idx))
        return false;

      if (m_ParIdx.CheckIdx(idx, 1))
        return m_InputNode.TrySetParam(idx, value, isDelta);

      return m_OutputNode.TrySetParam(idx, value, isDelta);
    }

    /// <summary>
    /// Tries to update parameters of the network and passes it down to all sublayers
    /// WARNING: override this method carefully!
    /// Do not use base. as base methods operate with base class index which may differ from exact class index
    /// and so it will return wrong results. Override this method COMPLETELY or not override at all
    /// </summary>
    /// <param name="pars">Parameter values to update</param>
    /// <param name="isDelta">Is the values are exact or just deltas to existing ones</param>
    /// <param name="cursor">Start position in parameter vector</param>
    /// <returns>True is operation succeeded, false otherwise (bad parameter vector unexisted indices etc.)</returns>
    public override bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (pars == null || pars.Length < cursor)
        return false;

      var success = m_InputNode.TryUpdateParams(pars, isDelta, ref cursor);
      if (!success) return false;

      return m_OutputNode.TryUpdateParams(pars, isDelta, ref cursor);
    }

    protected override double DoGetParam(int idx)
    {
      return 0;
    }

    protected override void DoSetParam(int idx, double value, bool isDelta)
    {
    }

    protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
    {
    }
  }
}
