using System;
using System.Collections.Generic;

namespace ML.Core.ComputingNetworks
{
  /// <summary>
  /// Less-generic shortcut version of SequenceNode<TIn, TOut, THidNode>
  ///
  ///  TIn -> TIn -> TIn -> ... -> TIn -> TOut
  ///
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  public class SequenceOutputNode<TIn, TOut> : SequenceOutputNode<TIn, TOut, ComputingNode<TIn, TIn>>
  {
  }

  /// <summary>
  /// Represents computing node joined from set of others -
  /// a node that sequentially tunells input through set of layers and finally through output node
  ///
  ///  TIn -> TIn -> TIn -> ... -> TIn -> TOut
  ///
  /// </summary>
  /// <typeparam name="TIn">Input object type</typeparam>
  /// <typeparam name="TOut">Output object type</typeparam>
  /// <typeparam name="THidNode">Type of hidden nodes</typeparam>
  public class SequenceOutputNode<TIn, TOut, THidNode> : ComputingNode<TIn, TOut>
    where THidNode : ComputingNode<TIn, TIn>
  {
    private ParamMultiIdx m_ParIdx;
    private ComputingNode<TIn, TOut> m_OutputNode;
    private THidNode[] m_HiddenNodes;


    public SequenceOutputNode()
    {
      m_HiddenNodes = new THidNode[0];
    }

    public override int ParamCount { get { return 0; } }
    public IComputingNode<TIn, TOut> OutputNode { get { return m_OutputNode; } }
    public THidNode[] HiddenNodes { get { return m_HiddenNodes; } }

    public void AddHiddenNode(THidNode hiddenNode)
    {
      if (hiddenNode==null)
        throw new MLException("Node can not be null");

      var len = m_HiddenNodes.Length;
      var hiddenNodes = new THidNode[len+1];
      for (int i=0; i<len; i++)
        hiddenNodes[i] = m_HiddenNodes[i];
      hiddenNodes[len] = hiddenNode;

      m_HiddenNodes = hiddenNodes;
    }

    public void SetOutputNode(ComputingNode<TIn, TOut> outputNode)
    {
      if (outputNode==null)
        throw new MLException("Node can not be null");

      m_OutputNode = outputNode;
    }

    public override TOut Calculate(TIn input)
    {
      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
        input = m_HiddenNodes[i].Calculate(input);

      return m_OutputNode.Calculate(input);
    }

    /// <summary>
    /// Builds node before use (build search index etc.)
    /// </summary>
    internal override void DoBuild(bool buildIndex)
    {
      if (m_OutputNode==null)
        throw new MLException("Output node has not been set");

      if (buildIndex) BuildIndex(0);

      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
        m_HiddenNodes[i].DoBuild(false);

      m_OutputNode.DoBuild(false);
    }

    /// <summary>
    /// Builds fast search index
    /// </summary>
    /// <param name="startIdx">Index start value</param>
    /// <returns>Index end value</returns>
    internal override int BuildIndex(int startIdx)
    {
      if (m_OutputNode==null)
        throw new MLException("Output node has not been set");

      var len = m_HiddenNodes.Length;
      var idxs = new int[len+2];
      idxs[0] = startIdx;

      var endIdx = startIdx;
      for (int i=0; i<len; i++)
      {
        endIdx = m_HiddenNodes[i].BuildIndex(endIdx);
        idxs[i+1]=endIdx;
      }

      endIdx = m_OutputNode.BuildIndex(endIdx);
      idxs[len+1] = endIdx;

      m_ParIdx = new ParamMultiIdx(idxs);

      return endIdx;
    }

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

      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
      {
        if (m_ParIdx.CheckIdx(idx, i+1))
          return m_HiddenNodes[i].TryGetParam(idx, out value);
      }

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

      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
      {
        if (m_ParIdx.CheckIdx(idx, i+1))
          return m_HiddenNodes[i].TrySetParam(idx, value, isDelta);
      }

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

      var len = m_HiddenNodes.Length;
      for (int i=0; i<len; i++)
      {
        var success = m_HiddenNodes[i].TryUpdateParams(pars, isDelta, ref cursor);
        if (!success) return false;
      }

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
