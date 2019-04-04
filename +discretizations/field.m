%
% superclass for all fields
%
% author: Martin F. Schiffner
% date: 2019-01-22
% modified: 2019-03-29
%
classdef field < discretizations.signal_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties

        % independent properties
        grid_FOV ( 1, 1 ) discretizations.grid

        % dependent properties
        size_bytes ( 1, 1 ) physical_values.byte	% memory consumption

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = field( axes, grids_FOV, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.grid
            if ~isa( grids_FOV, 'discretizations.grid' )
                errorStruct.message     = 'grids_FOV must be discretizations.grid!';
                errorStruct.identifier	= 'field:NoRegularGrid';
                error( errorStruct );
            end

            % specify default samples
            if nargin < 3
                samples = cell( size( axes ) );
                for index_object = 1:numel( axes )
                    samples{ index_object } = zeros( [ grids_FOV( index_object ).N_points, abs( axes( index_object ) ) ] );
                    if isa( grids_FOV( index_object ), 'discretizations.grid_regular' )
                        samples{ index_object } = reshape( samples{ index_object }, [ grids_FOV( index_object ).N_points_axis, abs( axes( index_object ) ) ] );
                    end
                end
            end

            % ensure cell array for samples
            if ~iscell( samples )
                samples = { samples };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( axes, grids_FOV );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.signal_matrix( axes, samples );

            %--------------------------------------------------------------
            % 3.) set independent and dependent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                % ensure correct size of grids_FOV
                if grids_FOV( index_object ).N_points ~= objects( index_object ).N_signals
                    errorStruct.message = sprintf( 'Number of grid points in grids_FOV( %d ) must equal the number of signals %d!', index_object, objects( index_object ).N_signals );
                    errorStruct.identifier = 'field:GridSizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).grid_FOV = grids_FOV( index_object );

                % compute memory consumption
                objects( index_object ).size_bytes = data_volume( objects( index_object ) );

            end % for index_object = 1:numel( objects )

        end % function objects = field( axes, grids_FOV, samples )

        %------------------------------------------------------------------
        % get_cell
        %------------------------------------------------------------------
        function objects = get_cell( fields )

            for index_object = 1:numel( fields )

                
            end
            % number of discrete frequencies
            N_dimensions_act = ndims( samples{ index_object } );
            size_act = size( samples{ index_object } );
            N_samples_f = size_act( end );
            N_points = prod( size_act( 1:(end - 1) ) );
            firstdims = repmat( {':'}, 1, N_dimensions_act - 1 );

            % initialize field values with zeros
            objects( index_object ).values = cell( 1, N_samples_f );
            for index_f = 1:N_samples_f
                objects( index_object ).values{ index_f } = samples{ index_object }( firstdims{ : }, index_f );
            end
        end

        %------------------------------------------------------------------
        % shift fields
        %------------------------------------------------------------------
        function objects_out = shift( objects_in )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % TODO: symmetric grid

            %--------------------------------------------------------------
            % 2.) shift spatial transfer functions
            %--------------------------------------------------------------

        end % function objects_out = shift( objects_in )

        %------------------------------------------------------------------
        % show
        %------------------------------------------------------------------
        function hdl = show( fields )

            %--------------------------------------------------------------
            % 1.) display fields
            %--------------------------------------------------------------
            hdl = zeros( size( fields ) );
            for index_object = 1:numel( fields )

                N_samples_f = numel( fields( index_object ).values );
                index_ctr = round( N_samples_f / 2 );

                hdl( index_object ) = figure( index_object );

                % check number of spatial dimensions
                switch ndims( fields( index_object ).values{ 1 } )

                    case 2

                        %--------------------------------------------------
                        % a) two-dimensional Euclidean space
                        %--------------------------------------------------
                        subplot( 2, 3, 1);
                        imagesc( abs( fields( index_object ).values{ 1 } ) );
                        subplot( 2, 3, 2);
                        imagesc( abs( fields( index_object ).values{ index_ctr } ) );
                        subplot( 2, 3, 3);
                        imagesc( abs( fields( index_object ).values{ end } ) );
                        subplot( 2, 3, 4);
                        imagesc( angle( fields( index_object ).values{ 1 } ) );
                        subplot( 2, 3, 5);
                        imagesc( angle( fields( index_object ).values{ index_ctr } ) );
                        subplot( 2, 3, 6);
                        imagesc( angle( fields( index_object ).values{ end } ) );

                    case 3

                        %--------------------------------------------------
                        % b) three-dimensional Euclidean space
                        %--------------------------------------------------
                        subplot( 3, 3, 1);
                        imagesc( abs( squeeze( fields( index_object ).values{ 1 }( :, 1, :) ) ) );
                        subplot( 3, 3, 2);
                        imagesc( abs( squeeze( fields( index_object ).values{ index_ctr }( :, 1, :) ) ) );
                        subplot( 3, 3, 3);
                        imagesc( abs( squeeze( fields( index_object ).values{ end }( :, 1, :) ) ) );
                        subplot( 3, 3, 4);
                        imagesc( abs( squeeze( fields( index_object ).values{ 1 }( :, :, 1) ) ) );
                        subplot( 3, 3, 5);
                        imagesc( abs( squeeze( fields( index_object ).values{ index_ctr }( :, :, 1) ) ) );
                        subplot( 3, 3, 6);
                        imagesc( abs( squeeze( fields( index_object ).values{ end }( :, :, 1) ) ) );
                        subplot( 3, 3, 7);
                        imagesc( abs( squeeze( fields( index_object ).values{ 1 }( 1, :, :) ) ) );
                        subplot( 3, 3, 8);
                        imagesc( abs( squeeze( fields( index_object ).values{ index_ctr }( 1, :, :) ) ) );
                        subplot( 3, 3, 9);
                        imagesc( abs( squeeze( fields( index_object ).values{ end }( 1, :, :) ) ) );

                    otherwise

                        %--------------------------------------------------
                        % c) dimensionality not implemented
                        %--------------------------------------------------
                        errorStruct.message     = 'Number of dimensions not implemented!';
                        errorStruct.identifier	= 'show:UnknownDimensions';
                        error( errorStruct );

                end % switch ndims( fields( index_object ).values{ 1 } )
                
            end % for index_object = 1:numel( fields )

        end % function hdl = show( fields )

	end % methods

end % classdef field < discretizations.signal_matrix
