%
% abstract superclass for all spatial anti-aliasing filters
%
% author: Martin F. Schiffner
% date: 2019-07-11
% modified: 2020-03-09
%
classdef (Abstract) anti_aliasing_filter

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing_filter( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'anti_aliasing_filter:NoRowVector';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create spatial anti-aliasing filters
            %--------------------------------------------------------------
            % repeat default spatial anti-aliasing filter
            objects = repmat( objects, size );

        end % function objects = anti_aliasing_filter( size )

        %------------------------------------------------------------------
        % apply spatial anti-aliasing filters
        %------------------------------------------------------------------
        function hs_transfer = apply( filters, setups, hs_transfer, indices_element )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.anti_aliasing_filter
            if ~isa( filters, 'scattering.anti_aliasing_filters.anti_aliasing_filter' )
                errorStruct.message = 'filters must be scattering.anti_aliasing_filters.anti_aliasing_filter!';
                errorStruct.identifier = 'apply:NoSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'apply:NoSetups';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.transducers.array_planar_regular_orthogonal
            indicator = cellfun( @( x ) ~isa( x, 'scattering.sequences.setups.transducers.array_planar_regular_orthogonal' ), { setups.xdc_array } );
            if any( indicator( : ) )
                errorStruct.message = 'setups.xdc_array must be scattering.sequences.setups.transducers.array_planar_regular_orthogonal!';
                errorStruct.identifier = 'apply:NoOrthogonalRegularPlanarArrays';
                error( errorStruct );
            end

            % ensure class processing.field
            if ~isa( hs_transfer, 'processing.field' )
                errorStruct.message = 'hs_transfer must be processing.field!';
                errorStruct.identifier = 'apply:NoFields';
                error( errorStruct );
            end

            % ensure nonempty indices_element
            if nargin < 4 || isempty( indices_element )
                indices_element = 1;
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( filters, setups, hs_transfer, indices_element );

            %--------------------------------------------------------------
            % 2.) apply spatial anti-aliasing filters
            %--------------------------------------------------------------
            % iterate spatial anti-aliasing filters
            for index_object = 1:numel( filters )

                %----------------------------------------------------------
                % a) apply spatial anti-aliasing filter (scalar)
                %----------------------------------------------------------
                hs_transfer( index_object ) = apply_scalar( filters( index_object ), setups( index_object ), hs_transfer( index_object ), indices_element( index_object ) );

            end % for index_object = 1:numel( filters )

        end % function hs_transfer = apply( filters, setups, hs_transfer, indices_element )

        %------------------------------------------------------------------
        % compute filter samples
        %------------------------------------------------------------------
        function samples = compute_samples( filters, flags )
% TODO: return fields
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.anti_aliasing_filter
            if ~isa( filters, 'scattering.anti_aliasing_filters.anti_aliasing_filter' )
                errorStruct.message = 'filters must be scattering.anti_aliasing_filters.anti_aliasing_filter!';
                errorStruct.identifier = 'compute_samples:NoSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            % ensure class processing.field
            if ~isa( flags, 'processing.field' )
                errorStruct.message = 'flags must be processing.field!';
                errorStruct.identifier = 'compute_samples:NoFields';
                error( errorStruct );
            end

            % multiple filters / single flags
            if ~isscalar( filters ) && isscalar( flags )
                flags = repmat( flags, size( filters ) );
            end

            % single filters / multiple flags
            if isscalar( filters ) && ~isscalar( flags )
                filters = repmat( filters, size( flags ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( filters, flags );

            %--------------------------------------------------------------
            % 2.) compute filter samples
            %--------------------------------------------------------------
            % specify cell array for samples
            samples = cell( size( filters ) );

            % iterate spatial anti-aliasing filters
            for index_filter = 1:numel( filters )

                % compute filter samples (scalar)
                samples{ index_filter } = compute_samples_scalar( filters( index_filter ), flags( index_filter ) );

            end % for index_filter = 1:numel( filters )

            % avoid cell array for single filters
            if isscalar( filters )
                samples = samples{ 1 };
            end

        end % function samples = compute_samples( filters, flags )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        strs_out = string( filters )

	end % methods (Abstract)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % apply spatial anti-aliasing filter (scalar)
        %------------------------------------------------------------------
        h_transfer = apply_scalar( filter, setup, h_transfer, index_element )

        %------------------------------------------------------------------
        % compute filter samples (scalar)
        %------------------------------------------------------------------
        samples = compute_samples_scalar( filter, flags )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) anti_aliasing_filter
