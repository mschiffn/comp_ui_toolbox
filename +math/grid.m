%
% superclass for all grids
%
% author: Martin F. Schiffner
% date: 2019-03-21
% modified: 2019-05-03
%
classdef grid

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties
% TODO: access rights?
        % independent properties
        positions ( :, : ) physical_values.length	% discrete positions of the grid points

        % dependent properties
        N_dimensions ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 2	% number of dimensions (1)
        N_points ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 16384	% total number of grid points (1)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = grid( positions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for no input arguments
            if nargin == 0
                return;
            end

            % ensure cell array for positions
            if ~iscell( positions )
                positions = { positions };
            end

            %--------------------------------------------------------------
            % 2.) create grids
            %--------------------------------------------------------------
            objects = repmat( objects, size( positions ) );

            % set independent and dependent properties
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).positions = positions{ index_object };

                % set dependent properties
                objects( index_object ).N_dimensions = size( objects( index_object ).positions, 2 );
                objects( index_object ).N_points = size( objects( index_object ).positions, 1 );

            end % for index_object = 1:numel( objects )

        end % function objects = grid( positions )

        %------------------------------------------------------------------
        % mutual differences
        %------------------------------------------------------------------
        function differences = mutual_differences( grids_1, grids_2, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.grid
            if ~( isa( grids_1, 'math.grid' ) && isa( grids_2, 'math.grid' ) )
                errorStruct.message     = 'Both arguments must be math.grid!';
                errorStruct.identifier	= 'mutual_differences:NoGrids';
                error( errorStruct );
            end

            % ensure nonempty indices_1
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                indices_1 = varargin{ 1 };
            else
                indices_1 = cell( size( grids_1 ) );
                for index_object = 1:numel( grids_1 )
                    indices_1{ index_object } = (1:grids_1( index_object ).N_points);
                end
            end

            % ensure cell array for indices_1
            if ~iscell( indices_1 )
                indices_1 = { indices_1 };
            end

            % ensure nonempty indices_2
            if nargin >= 4 && ~isempty( varargin{ 2 } )
                indices_2 = varargin{ 2 };
            else
                indices_2 = cell( size( grids_2 ) );
                for index_object = 1:numel( grids_2 )
                    indices_2{ index_object } = (1:grids_2( index_object ).N_points);
                end
            end

            % ensure cell array for indices_2
            if ~iscell( indices_2 )
                indices_2 = { indices_2 };
            end

            % multiple grids_1 / single indices_1
            if ~isscalar( grids_1 ) && isscalar( indices_1 )
                indices_1 = repmat( indices_1, size( grids_1 ) );
            end

            % multiple grids_2 / single indices_2
            if ~isscalar( grids_2 ) && isscalar( indices_2 )
                indices_2 = repmat( indices_2, size( grids_2 ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( grids_1, grids_2, indices_1, indices_2 );

            %--------------------------------------------------------------
            % 2.) compute mutual differences for each pair of grids
            %--------------------------------------------------------------
            % specify cell array
            differences = cell( size( grids_1 ) );

            % iterate pairs of grids
            for index_object = 1:numel( grids_1 )

                % maximum number of dimensions
                N_dimensions_max = max( [ grids_1( index_object ).N_dimensions, grids_2( index_object ).N_dimensions ] );

                % numbers of relevant grid points
                N_points_1 = numel( indices_1{ index_object } );
                N_points_2 = numel( indices_2{ index_object } );

                % inflate relevant positions to correct dimension
% TODO: potential unit conflict with zeros!
                positions_1 = [ grids_1( index_object ).positions( indices_1{ index_object }, : ), zeros( N_points_1, N_dimensions_max - grids_1( index_object ).N_dimensions ) ];
                positions_2 = [ grids_2( index_object ).positions( indices_2{ index_object }, : ), zeros( N_points_2, N_dimensions_max - grids_2( index_object ).N_dimensions ) ];

                % reshape relevant positions to compute mutual distances
                positions_1 = repmat( reshape( positions_1, [ N_points_1, 1, N_dimensions_max ] ), [ 1, N_points_2, 1] );
                positions_2 = repmat( reshape( positions_2, [ 1, N_points_2, N_dimensions_max ] ), [ N_points_1, 1, 1] );

                % compute differences
                differences{ index_object } = positions_1 - positions_2;

            end % for index_object = 1:numel( grids_1 )

            % avoid cell array for single pair of grids
            if isscalar( grids_1 )
                differences = differences{ 1 };
            end

        end % function differences = mutual_differences( grids_1, grids_2, varargin )

        %------------------------------------------------------------------
        % mutual distances
        %------------------------------------------------------------------
        function [ D, differences ] = mutual_distances( grids_1, grids_2, varargin )

            %--------------------------------------------------------------
            % 1.) mutual differences
            %--------------------------------------------------------------
            differences = mutual_differences( grids_1, grids_2, varargin{ : } );

            % ensure cell array for differences
            if ~iscell( differences )
                differences = { differences };
            end

            %--------------------------------------------------------------
            % 2.) compute mutual distances for each pair of grids
            %--------------------------------------------------------------
            % specify cell array
            D = cell( size( differences ) );

            % iterate pairs of grids
            for index_object = 1:numel( differences )

                % compute l2-norms
                D{ index_object } = sqrt( sum( differences{ index_object }.^2, 3 ) );

            end % for index_object = 1:numel( differences )

            % avoid cell arrays for single pair of grids
            if isscalar( grids_1 )
                D = D{ 1 };
                differences = differences{ 1 };
            end

        end % function [ D, differences ] = mutual_distances( grids_1, grids_2, varargin )

        %------------------------------------------------------------------
        % mutual unit vectors
        %------------------------------------------------------------------
        function [ e_1_minus_2, D ] = mutual_unit_vectors( grids_1, grids_2, varargin )

            %--------------------------------------------------------------
            % 1.) mutual distances
            %--------------------------------------------------------------
            [ D, differences ] = mutual_distances( grids_1, grids_2, varargin{ : } );

            % ensure cell array for D
            if ~iscell( D )
                D = { D };
            end

            % ensure cell array for differences
            if ~iscell( differences )
                differences = { differences };
            end

            %--------------------------------------------------------------
            % 2.) compute mutual unit vectors for each pair of grids
            %--------------------------------------------------------------
            % specify cell array
            e_1_minus_2 = cell( size( differences ) );

            % iterate pairs of grids
            for index_object = 1:numel( differences )

                % compute unit vectors
                e_1_minus_2{ index_object } = differences{ index_object } ./ D{ index_object };

            end % for index_object = 1:numel( differences )

            % avoid cell array for single pair of grids
            if isscalar( grids_1 )
                D = D{ 1 };
                e_1_minus_2 = e_1_minus_2{ 1 };
            end

        end % function [ e_1_minus_2, D ] = mutual_unit_vectors( grids_1, grids_2, varargin )

        %------------------------------------------------------------------
        % extract subset of grid points
        %------------------------------------------------------------------
        function grids_out = subgrid( grids_in, indices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices
            if ~iscell( indices )
                indices = { indices };
            end

            % multiple grids_in / single indices
            if ~isscalar( grids_in ) && isscalar( indices )
                indices = repmat( indices, size( grids_in ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( grids_in, indices );

            %--------------------------------------------------------------
            % 2.) extract grid points
            %--------------------------------------------------------------
            % specify cell array
            positions_act = cell( size( grids_in ) );

            % iterate grids
            for index_object = 1:numel( grids_in )
                positions_act{ index_object } = grids_in( index_object ).positions( indices{ index_object }, : );
            end % for index_object = 1:numel( grids_in )

            %--------------------------------------------------------------
            % 3.) create grids
            %--------------------------------------------------------------
            grids_out = math.grid( positions_act );

        end % function grids_out = subgrid( grids_in, indices )

    end % methods

end % classdef grid
