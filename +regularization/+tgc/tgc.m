%
% abstract superclass for all time gain compensation (TGC) options
%
% author: Martin F. Schiffner
% date: 2019-12-15
% modified: 2020-03-18
%
classdef (Abstract) tgc < regularization.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = tgc( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create TGC options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.template( size );

        end % function objects = tgc( size )

        %------------------------------------------------------------------
        % create linear transforms
        %------------------------------------------------------------------
        function [ LTs, LTs_measurement ] = get_LTs( tgcs, operators_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.tgc.tgc
            if ~isa( tgcs, 'regularization.tgc.tgc' )
                errorStruct.message = 'tgcs must be regularization.tgc.tgc!';
                errorStruct.identifier = 'get_LTs:NoTGCs';
                error( errorStruct );
            end

            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'get_LTs:NoOperatorsBorn';
                error( errorStruct );
            end

            % multiple tgcs / single operators_born
            if ~isscalar( tgcs ) && isscalar( operators_born )
                operators_born = repmat( operators_born, size( tgcs ) );
            end

            % single tgcs / multiple operators_born
            if isscalar( tgcs ) && ~isscalar( operators_born )
                tgcs = repmat( tgcs, size( operators_born ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( tgcs, operators_born );

            %--------------------------------------------------------------
            % 2.) create linear transforms
            %--------------------------------------------------------------
            % specify cell array for LTs
            LTs = cell( size( tgcs ) );
            LTs_measurement = cell( size( tgcs ) );

            % iterate TGCs
            for index_tgc = 1:numel( tgcs )

                % create linear transform (scalar)
                [ LTs{ index_tgc }, LTs_measurement{ index_tgc } ] = get_LT_scalar( tgcs( index_tgc ), operators_born( index_tgc ) );

            end % for index_tgc = 1:numel( tgcs )

            % concatenate vertically
            LTs = reshape( cat( 1, LTs{ : } ), size( operators_born ) );

            % avoid cell arrays for single TGC
            if isscalar( tgcs )
                LTs_measurement = LTs_measurement{ 1 };
            end

        end % function [ LTs, LTs_measurement ] = get_LTs( tgcs, operators_born )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        [ LT, LT_measurement ] = get_LT_scalar( tgc, operator_born )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) tgc < regularization.options.template
