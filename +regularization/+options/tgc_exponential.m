%
% superclass for all exponential time gain compensation (TGC) options
%
% author: Martin F. Schiffner
% date: 2019-12-19
% modified: 2020-01-15
%
classdef tgc_exponential < regularization.options.tgc

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        exponents ( :, 1 ) physical_values.frequency { mustBePositive, mustBeNonempty } = physical_values.hertz( 1 )
        decays_dB ( :, 1 ) double { mustBeNegative, mustBeNonempty } = -40

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = tgc_exponential( exponents, decays_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for exponents
            if ~iscell( exponents )
                exponents = { exponents };
            end

            % ensure cell array for exponents
            if ~iscell( decays_dB )
                decays_dB = { decays_dB };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( exponents, decays_dB );

            %--------------------------------------------------------------
            % 2.) create exponential TGC options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.tgc( size( exponents ) );

            % iterate exponential TGC options
            for index_object = 1:numel( objects )

                % property validation function ensures class physical_values.frequency for exponents{ index_object }
                % property validation function ensures nonempty negative doubles for decays_dB{ index_object }

                % multiple exponents{ index_object } / single decays_dB{ index_object }
                if ~isscalar( exponents{ index_object } ) && isscalar( decays_dB{ index_object } )
                    decays_dB{ index_object } = repmat( decays_dB{ index_object }, size( exponents{ index_object } ) );
                end

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( exponents{ index_object }, decays_dB{ index_object } );

                % set independent properties
                objects( index_object ).exponents = exponents{ index_object }( : );
                objects( index_object ).decays_dB = decays_dB{ index_object }( : );

            end % for index_object = 1:numel( objects )

        end % function objects = tgc_exponential( exponents, decays_dB )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( tgcs_exponential )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.tgc_exponential
            if ~isa( tgcs_exponential, 'regularization.options.tgc_exponential' )
                errorStruct.message = 'tgcs_exponential must be regularization.options.tgc_exponential!';
                errorStruct.identifier = 'string:NoOptionsTGCExponential';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initializse string array for strs_out
            strs_out = repmat( "", size( tgcs_exponential ) );

            % iterate exponential TGC options
            for index_object = 1:numel( tgcs_exponential )

                strs_out( index_object ) = sprintf( "%s", 'exponential' );

            end % for index_object = 1:numel( tgcs_exponential )

        end % function strs_out = string( tgcs_exponential )

	end % methods

end % classdef tgc_exponential < regularization.options.tgc