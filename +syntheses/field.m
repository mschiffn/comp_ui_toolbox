%
% superclass for all fields
%
% author: Martin F. Schiffner
% date: 2019-01-22
% modified: 2019-02-20
%
classdef field

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties

        % independent properties
        values % phasors of the acoustic value for each grid point in the FOV

        % dependent properties
        size_bytes % ( 1, 1 ) physical_values.memory % memory consumption (B)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = field( discretization )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no input argument
            if nargin == 0
                return;
            end

            % ensure class discretizations.spatiospectral
            if ~( isa( discretization, 'discretizations.spatiospectral' ) && numel( discretization ) == 1 )
                errorStruct.message     = 'discretization must be a single discretizations.spatiospectral!';
                errorStruct.identifier	= 'field:NoDiscretization';
                error( errorStruct );
            end

            % ensure class discretizations.spatial_grid
            if ~isa( discretization.space, 'discretizations.spatial_grid' )
                errorStruct.message     = 'discretization.space must be discretizations.spatial_grid!';
                errorStruct.identifier	= 'field:NoSpatialGrid';
                error( errorStruct );
            end

            % ensure class discretizations.spectral_points
            if ~isa( discretization.frequency, 'discretizations.spectral_points' )
                errorStruct.message     = 'discretization.frequency must be discretizations.spectral_points!';
                errorStruct.identifier	= 'field:NoSetDiscreteFrequency';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) construct objects
            %--------------------------------------------------------------
            % construct column vector of objects
            objects = repmat( objects, size( discretization.frequency ) );

            %--------------------------------------------------------------
            % 3.) initialize objects with zeros / compute memory consumption
            %--------------------------------------------------------------
            for index_object = 1:numel( discretization.frequency )

                % number of discrete frequencies
                N_samples_f = abs( discretization.frequency( index_object ) );
                N_points = discretization.space.grid_FOV.N_points;

                % initialize field values with zeros
                objects( index_object ).values = cell( 1, N_samples_f );
                for index_f = 1:N_samples_f
                    objects( index_object ).values{ index_f } = zeros( discretization.space.grid_FOV.N_points_axis( 2 ), discretization.space.grid_FOV.N_points_axis( 1 ) );
                end

                % compute memory consumption
                objects( index_object ).size_bytes = physical_values.memory( N_samples_f * N_points * 16 );

            end % for index_object = 1:numel( discretization.frequency )

        end % function objects = field( discretization )

        %------------------------------------------------------------------
        % show (overload display function)
        %------------------------------------------------------------------
        function hdl = show( objects )

            N_objects = size( objects, 1 );
            hdl = zeros( N_objects, 1 );
            for index_object = 1:N_objects

                N_samples_f = numel( objects( index_object ).values );
                index_ctr = round( N_samples_f / 2 );

                hdl( index_object ) = figure( index_object );
                subplot( 2, 3, 1);
                imagesc( abs( objects( index_object ).values{ 1 } ) );
                subplot( 2, 3, 2);
                imagesc( abs( objects( index_object ).values{ index_ctr } ) );
                subplot( 2, 3, 3);
                imagesc( abs( objects( index_object ).values{ end } ) );
                subplot( 2, 3, 4);
                imagesc( angle( objects( index_object ).values{ 1 } ) );
                subplot( 2, 3, 5);
                imagesc( angle( objects( index_object ).values{ index_ctr } ) );
                subplot( 2, 3, 6);
                imagesc( angle( objects( index_object ).values{ end } ) );
            end
        end % function hdl = show( objects )

	end % methods

end % classdef field
