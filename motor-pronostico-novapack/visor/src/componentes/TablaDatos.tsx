/**
 * DataGrid genérico: columnas deducidas de las filas, orden y filtro incluidos.
 * Es la tabla de la mayoría de las vistas; para casos con columnas a medida se
 * usa DataGrid directo.
 */

import { useMemo } from "react";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";

import { numero } from "../tema/paleta";

interface Props {
  filas: Record<string, unknown>[];
  alto?: number;
  decimales?: number;
  paginado?: number;
}

export default function TablaDatos({
  filas,
  alto = 420,
  decimales = 2,
  paginado = 25,
}: Props) {
  const columnas = useMemo<GridColDef[]>(() => {
    if (filas.length === 0) return [];
    return Object.keys(filas[0]).map((campo) => ({
      field: campo,
      headerName: campo.replace(/_/g, " "),
      flex: 1,
      minWidth: 110,
      type: typeof filas[0][campo] === "number" ? "number" : "string",
      valueFormatter: (valor: unknown) =>
        typeof valor === "number" && !Number.isInteger(valor)
          ? numero(valor, decimales)
          : (valor as string),
    }));
  }, [filas, decimales]);

  const conId = useMemo(
    () => filas.map((fila, indice) => ({ id: indice, ...fila })),
    [filas],
  );

  return (
    <div style={{ height: alto, width: "100%" }}>
      <DataGrid
        rows={conId}
        columns={columnas}
        density="compact"
        disableRowSelectionOnClick
        initialState={{
          pagination: { paginationModel: { pageSize: paginado } },
        }}
        pageSizeOptions={[paginado, 50, 100]}
        sx={{ bgcolor: "background.paper" }}
      />
    </div>
  );
}
