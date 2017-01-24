using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ML.Core;
using ML.Contracts;
using System.IO;

namespace ML.ConsoleTest
{
  public class Visualizer
  {
    private const string COMMA = ",";

    private float m_XMin;
    private float m_XMax;
    private float m_YMin;
    private float m_YMax;
    private int   m_XCnt;
    private int   m_YCnt;

    public Visualizer(DataWrapper data)
    {
      Data = data;
    }

    public readonly DataWrapper Data;

    public void Run(IAlgorithm algorithm)
    {
      var fn = string.Format("data/{0}-{1:yyyyMMdd-hhmmss}.csv", algorithm.ID, DateTime.Now);
      using (var file = File.Create(fn))
      using (var writer = new StreamWriter(file))
      {
        prepareCanvas();
        doCalc(algorithm, writer);
      }
    }

    private void prepareCanvas()
    {
      m_XMin = Data.Data.Min(d => d.Key[0]);
      m_XMax = Data.Data.Max(d => d.Key[0]);
      m_YMin = Data.Data.Min(d => d.Key[1]);
      m_YMax = Data.Data.Max(d => d.Key[1]);

      m_XCnt = 200;
      m_YCnt = 200;
    }

    private void doCalc(IAlgorithm algorithm, StreamWriter writer)
    {
      for (var k=0; k<Math.Max(m_XCnt, m_YCnt); k++)
      {
        var px = (k<m_XCnt) ? getPoint(k, 0)[0].ToString() : string.Empty;
        var py = (k<m_YCnt) ? getPoint(0, k)[1].ToString() : string.Empty;
        var pd = string.Empty;
        if (k<m_YCnt)
        {
          var data = new float[m_XCnt];
          for (int i=0; i<m_XCnt; i++)
          {
            var point = getPoint(i, k);
            var cls = algorithm.Classify(point);
            data[i] = cls.Value;
          }
          pd = string.Join(COMMA, data);
        }

        writer.WriteLine("{0},{1},{2}", px, py, pd);
      }
    }

    private Point getPoint(int i, int j)
    {
      var result = new Point(2);
      result[0] = m_XMin + (m_XMax-m_XMin)*i/(m_XCnt-1);
      result[1] = m_YMin + (m_YMax-m_YMin)*j/(m_YCnt-1);

      return result;
    }
  }
}
