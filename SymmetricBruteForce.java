import java.util.*;
public class SymmetricBruteForce {
    private static int[] toarr(List<Integer> L) {
        int[] A=new int[L.size()];
        for (int i=0; i<L.size(); i++) A[i]=L.get(i);
        return A;
    }
    private static int N, N2, N4, N6, PR, Z;
    private static String matstr(int[] A) {
        StringBuilder s=new StringBuilder();
        for (int r=0; r<N; r++) {
            String t=Arrays.toString(Arrays.copyOfRange(A,r*N,r*N+N));
            s.append(r==0?"[[":" [")
                    .append(t.substring(1,t.length()-1).replace(",",""))
                    .append(r==N-1?"]]":"] ")
                    .append("\n");
        }
        return s.substring(0,s.length()-1);
    }
    private static int[] T;
    private static long TIMESTEP;
    private static int[] permd(int[] m, int[] p) {
        int[] out=new int[N2];
        for (int i=0; i<N; i++) for (int j=0; j<N; j++) out[p[i]*N+p[j]]=m[i*N+j];
        return out;
    }
    private static void add_oprod(int[] ret, int[] A, int[] B, int[] C) {
        for (int a=0; a<N2; a++) if (A[a]!=0)
            for (int b=0; b<N2; b++) if (B[b]!=0)
                for (int c=0; c<N2; c++) if (C[c]!=0)
                    ret[a*N4+b*N2+c]+=A[a]*B[b]*C[c];
    }
    private static List<int[]> nnmats(int Z) {
        List<int[]> mats=new ArrayList<>();
        int[] att=new int[N2]; Arrays.fill(att,-1);
        while (att[N2-1]<2) {
            int nz=0;
            for (int v:att) if (v!=0) nz++;
            if (0<nz&&nz<=Z) mats.add(Arrays.copyOf(att,N2));
            att[0]++;
            for (int i=0; i<N2-1&&att[i]==2; i++) {
                att[i]=-1;
                att[i+1]++;
            }
        }
        return mats;
    }
    private static int SCRMEM;
    private static int G; private static int[] groupid; private static int[][] groups;
    private static List<TableArr> ptablearrs;
    private static int[] curvals;
    //private static Map<String,Info> mem;
    private static class TableArr {
        //an array of "table"s + another array carrying some information about the array
        //a "table" is a tuple (in this case, triplet) of arrays that describes a rank-1 tensor
        int[] info;
        int[][][] tables;
        public TableArr(int[] i, int[][][] p) {
            info=i;
            tables=p;
        }
    }
    private static class Info {
        int scr;
        TableArr decomp;
        public Info(int s, TableArr d) {
            scr=s; decomp=d;
        }
    }
    private static class Node {
        Info info;
        int R;
        int[] cidxs;
        public Node(Info i, int r) {
            info=i; R=r;
            cidxs=new int[2*R+1]; Arrays.fill(cidxs,-1);
        }
        public void setChild(int v, int c) {
            cidxs[v+R]=c;
        }
        public int childId(int v) {
            return cidxs[v+R];
        }
    }
    private static List<Node> trie;
    private static long dfs_st, dfs_mark, dfs_work, dfs_sols;
    private static int dfs_ptablei;
    private static void dfs(int gi, int diff) {
        long t=System.currentTimeMillis();
        if (t>=dfs_mark) {
            dfs_mark+=1000;
            System.out.println((t-dfs_st)+" "+dfs_ptablei+" "+dfs_work+" "+dfs_sols);
        }
        dfs_work++;
        if (gi==G) {
            Info I=new Info(diff,ptablearrs.get(dfs_ptablei));
            //try putting new key into trie
            boolean none=false;
            int i=0;
            for (int g=0; g<G; g++) {
                int v=curvals[g];
                Node n=trie.get(i);
                if (n.childId(v)==-1) {
                    trie.add(new Node(null,g==G-1?0: groups[g+1].length));
                    n.setChild(v,trie.size()-1);
                    none=true;
                }
                i=n.childId(v);
            }
            if (none) {
                trie.get(i).info=I;
                dfs_sols++;
            }
            return;
        }
        int R=groups[gi].length;
        for (int v=-R; v<=R; v++) {
            curvals[gi]=v;
            //check score
            int ndiff=diff+(ptablearrs.get(dfs_ptablei).info[gi]!=curvals[gi]?groups[gi].length:0);
            if (ndiff<=SCRMEM) dfs(gi+1,ndiff);
        }
    }
    private static List<int[]> perms(int n) {
        if (n==1) return new ArrayList<>(Collections.singletonList(new int[] {0}));
        List<int[]> out=new ArrayList<>(), help=perms(n-1);
        for (int[] h:help)
            for (int i=0; i<n; i++) {
                int[] p=new int[n];
                System.arraycopy(h,0,p,0,i);
                p[i]=n-1;
                System.arraycopy(h,i,p,i+1,n-1-i);
                out.add(p);
            }
        return out;
    }
    public static void main(String[] args) {
        TIMESTEP=10_000;
        //n=size of MM tensor
        //pr=max # prefix matrices to use in tensor decomp
        //z=max
        N=3; PR=5; Z=9; SCRMEM=9;
        N2=N*N; N4=N2*N2; N6=N4*N2;
        System.out.printf("N=%d,PR=%d,Z=%d%n",N,PR,Z);
        System.out.println("only consider decomps w/ scr <= "+SCRMEM);
        List<int[]> Ps=perms(N);
        groups=new int[N6][]; G=0; groupid=new int[N6]; {
            boolean[] open=new boolean[N6]; Arrays.fill(open,true);
            for (int i=0; i<N6; i++) if (open[i]) {
                int[] coords={i/N4,(i/N2)%N2,i%N2};
                List<Integer> g=new ArrayList<>();
                for (int cyc=0; cyc<3; cyc++) for (int[] P:Ps) {
                    int[] nc=new int[3];
                    for (int a=0; a<3; a++) {
                        int rc=coords[(a+cyc)%3], r=rc/N, c=rc%N;
                        nc[a]=P[r]*N+P[c];
                    }
                    int ci=nc[0]*N4+nc[1]*N2+nc[2];
                    if (open[ci]) g.add(ci);
                    open[ci]=false;
                    groupid[ci]=G;
                }
                groups[G++]=toarr(g);
            }
            groups=Arrays.copyOf(groups,G);
        }
        //for (int[] g:idxgroups) System.out.println(Arrays.toString(g));
        T=new int[N6];
        for (int i=0; i<N; i++) for (int j=0; j<N; j++) for (int k=0; k<N; k++)
            T[(i*N+j)*N4+(j*N+k)*N2+(k*N+i)]=1;
        List<int[]> mats=nnmats(Z);
        System.out.println("# mats="+mats.size());
        //TODO: TRY OTHER PREFIX DECOMPS
        ptablearrs=new ArrayList<>(); {
            List<int[]> premats=new ArrayList<>();
            for (int[] m:nnmats(N2)) {
                boolean good=true;
                for (int[] p:Ps) if (!Arrays.equals(m,permd(m,p))) {
                    good=false;
                    break;
                }
                if (good) premats.add(m);
            }
            System.out.println("# premats="+premats.size());
            Map<String,TableArr> tmp=new HashMap<>();
            for (int mask=0; mask<(1<<premats.size()); mask++) {
                List<Integer> idxs=new ArrayList<>();
                for (int b=0; b<premats.size(); b++) if (((mask>>b)&1)!=0) idxs.add(b);
                if (idxs.size()<=PR) {
                    int[] info=new int[G]; {
                        int[] ret=new int[N6];
                        for (int i:idxs) {
                            int[] m=premats.get(i);
                            add_oprod(ret,m,m,m);
                        }
                        for (int i=0; i<N6; i++) ret[i]=T[i]-ret[i];
                        for (int gi=0; gi<G; gi++) info[gi]=ret[groups[gi][0]];
                    }
                    String c=Arrays.toString(info);
                    if (!tmp.containsKey(c)||idxs.size()<tmp.get(c).tables.length) {
                        int[][][] tables=new int[idxs.size()][][];
                        for (int i=0; i<idxs.size(); i++) {
                            int[] m=premats.get(idxs.get(i));
                            tables[i]=new int[][] {m,m,m};
                        }
                        tmp.put(c,new TableArr(info,tables));
                    }
                }
            }
            for (String s:tmp.keySet()) ptablearrs.add(tmp.get(s));
            System.out.println("# psums="+ ptablearrs.size());
        }
        //mem=new HashMap<>();
        trie=new ArrayList<>(Collections.singletonList(new Node(null, groups[0].length))); {
            curvals=new int[G];
            dfs_st=System.currentTimeMillis(); dfs_mark=dfs_st+1000;
            dfs_work=0; dfs_sols=0;
            System.out.println("DFS:");
            for (dfs_ptablei=0; dfs_ptablei<ptablearrs.size(); dfs_ptablei++) dfs(0,0);
            long t=System.currentTimeMillis();
            System.out.println((t-dfs_st)+" "+dfs_ptablei+" "+dfs_work+" "+dfs_sols);
        }
        List<int[]> posvmats=new ArrayList<>();
        //normalize signs of mats by forcing the first nonzero element along row-major order positive
        for (int[] m:mats) {
            int bv=0;
            for (int v:m) if (v!=0) {
                bv=v;
                break;
            }
            if (bv==0) throw new RuntimeException("all 0s matrix encountered");
            if (bv>0) posvmats.add(m);
        }
        System.out.println("# \"positive\" mats="+posvmats.size());
        Map<String,String> canonicalForm=new HashMap<>();
        List<int[]> canonicals=new ArrayList<>();
        for (int[] mat:mats) {
            String s=Arrays.toString(mat);
            String bs=null;
            int[] nmat=new int[N2]; for (int i=0; i<N2; i++) nmat[i]=-mat[i];
            for (int[] m:new int[][] {mat,nmat}) for (int[] P:Ps) {
                String ns=Arrays.toString(permd(m,P));
                if (bs==null||ns.compareTo(bs)<0) bs=ns;
            }
            canonicalForm.put(s,bs);
            if (s.equals(bs)) canonicals.add(mat);
        }
        System.out.println("# canonical mats="+canonicals.size());
        long cnt=0, st=System.currentTimeMillis(), mark=st+TIMESTEP;
        int bscr=SCRMEM+1;
        long trie_work=0;
        System.out.println("SEARCH:");
        for (int[] ma:canonicals) {
            String ca=canonicalForm.get(Arrays.toString(ma));
            List<int[]> Bs=new ArrayList<>(), Cs=new ArrayList<>();
            for (int[] m:posvmats) if (ca.compareTo(canonicalForm.get(Arrays.toString(m)))<0) Bs.add(m);
            for (int[] m:mats) if (ca.compareTo(canonicalForm.get(Arrays.toString(m)))<0) Cs.add(m);
            for (int[] mb:Bs) for (int[] mc:Cs) {
                Info binfo=new Info(Integer.MAX_VALUE,null); {
                    int i=0;
                    for (int g=0; g<G&&i>-1; g++, trie_work++) {
                        int v=0;
                        for (int ti:groups[g]) v+=ma[ti/N4]*mb[(ti/N2)%N2]*mc[ti%N2];
                        i=trie.get(i).childId(v);
                    }
                    if (i>-1) binfo=trie.get(i).info;
                }
                if (binfo.scr<bscr) {
                    bscr=binfo.scr;
                    System.out.println("bscr="+bscr);
                    System.out.println("prefix mats:");
                    for (int[][] table:binfo.decomp.tables) {
                        String[] lines=matstr(table[0]).split("\n");
                        for (int t=1; t<3; t++) {
                            String[] nlines=matstr(table[t]).split("\n");
                            String[] ret=new String[Math.max(lines.length,nlines.length)];
                            for (int i=0; i<ret.length; i++)
                                ret[i]=(i<lines.length?lines[i]:"")+(i==1?" * ":"   ")+(i<nlines.length?nlines[i]:"");
                            lines=ret;
                        }
                        for (String l:lines) System.out.println(l);
                    }
                    System.out.println("main mats:");
                    for (int[] m:new int[][] {ma,mb,mc}) System.out.println(matstr(m));
                    System.out.println();
                }
                cnt++;
                long t=System.currentTimeMillis();
                if (t>=mark) {
                    System.out.println((t-st)+" "+cnt+" "+trie_work);
                    mark+=TIMESTEP;
                }
            }
        }
        long t=System.currentTimeMillis();
        System.out.println((t-st)+" "+cnt+" "+trie_work);
        System.out.println("bscr="+bscr);
    }
}
