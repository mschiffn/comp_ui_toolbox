% superclass for all orthotopes using grids
%
% author: Martin F. Schiffner
% date: 2019-08-20
% modified: 2019-08-22
%
classdef orthotope_grid < geometry.orthotope

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        grid ( 1, 1 ) math.grid { mustBeNonempty } = math.grid

    end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthotope_grid( grids, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.grid
            if ~isa( grids, 'math.grid' )
                errorStruct.message = 'grids must be math.grid!';
                errorStruct.identifier = 'orthotope_grid:NoGrids';
                error( errorStruct );
            end

            % superclass ensures valid math.interval

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( grids, varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create orthotopes using grids
            %--------------------------------------------------------------
            % constructor of superclass
            objects@geometry.orthotope( varargin{ : } );

            % iterate orthotopes using grids
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).grid = grids( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = orthotope_grid( grids, varargin )

	end % methods

end % classdef orthotope_grid < geometry.orthotope
