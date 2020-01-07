%
% superclass for all curve time gain compensation (TGC) options
%
% author: Martin F. Schiffner
% date: 2019-12-16
% modified: 2020-01-03
%
classdef tgc_curve < regularization.options.tgc

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        curve ( 1, 1 ) %tgc.curve { mustBeNonempty } = regularization.tgc.exponential
        decay_dB ( 1, 1 ) double { mustBeNegative, mustBeNonempty } = -40

	end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = tgc_curve( curves, decays_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation function ensures class regularization.tgc.curve for curves
            % property validation function ensures nonempty negative doubles for decays_dB

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( curves, decays_dB );

            %--------------------------------------------------------------
            % 2.) create curve TGC options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.tgc( size( curves ) );

            % iterate curve TGC options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).curve = curves( index_object );
                objects( index_object ).decay_dB = decays_dB( index_object );

            end

        end % function objects = tgc_curve( varargin )

	end % methods

end % classdef tgc_curve < regularization.options.tgc
